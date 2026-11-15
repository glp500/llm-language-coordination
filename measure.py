"""Reused NVML/psutil measurements, CodeCarbon totals and separate EcoLogits estimates.

Importing set_phase inside the model container needs only the standard library.
EcoLogits stays in the host Conda environment, outside frozen GPU dependencies.
Phase boundaries are acknowledged before child work starts. GPU energy comes
from NVML counters, not duration-based allocation of whole-command energy.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def estimate_request(call, profile=None):
    """Use EcoLogits' published estimator; a wire protocol is not a deployment."""
    if call.get("completion_tokens") is None:
        return {"status": "usage_unavailable", "impacts": None}
    if not profile:
        return {"status": "unsupported_model_or_deployment", "impacts": None}
    if set(profile) - {"provider", "model", "scope", "electricity_mix_zone"} or profile.get("scope") not in {
        "deployment",
        "reference",
    }:
        raise ValueError("Impact profile needs provider/model, scope deployment/reference and optional electricity_mix_zone")
    if profile["scope"] == "deployment" and call.get("returned_model") and call["returned_model"] != profile["model"]:
        raise ValueError("Deployment impact model differs from the returned model; use an explicit reference scenario")
    try:
        from ecologits.tracers.utils import llm_impacts

        result = llm_impacts(
            profile["provider"],
            profile["model"],
            call["completion_tokens"],
            call["backend_batch_end_monotonic"] - call["backend_batch_start_monotonic"],
            profile.get("electricity_mix_zone"),
        )
    except ImportError:
        return {"status": "dependency_unavailable", "impacts": None}
    payload = result.model_dump(mode="json")
    unsupported = any(e.get("code") == "model-not-registered" for e in payload.get("errors") or [])
    return {
        "status": "unsupported_model_or_deployment" if unsupported else "estimation_error" if result.has_errors else "estimated",
        "impacts": payload,
    }


def estimate_saved_requests(source, output, profile=None, job_id=None):
    """Estimate each recorded request once; unavailable model profiles stay unavailable."""
    from collections import Counter

    counts = Counter()
    with (output / "impacts.jsonl").open("x") as stream:
        for path in sorted(source.rglob("requests.jsonl")):
            for line in path.read_text().splitlines():
                call = json.loads(line)
                if job_id is not None and call.get("slurm_job_id") != job_id:
                    continue
                value = estimate_request(call, profile)
                counts[value["status"]] += 1
                stream.write(json.dumps({"source": str(path.relative_to(source)), "call": call["call"], **value}) + "\n")
    write_json(
        output / "impacts-summary.json",
        {
            "estimator": "EcoLogits",
            "version": version("ecologits"),
            "requests": sum(counts.values()),
            "statuses": dict(counts),
            "scope": "request_estimates_not_measured_hardware_energy",
        },
    )
    return counts


def grid_factor(zone="NLD"):
    """Official EcoLogits electricity coefficient; never assume one if unavailable."""
    try:
        from ecologits.electricity_mix_repository import electricity_mixes

        mix = electricity_mixes.find_electricity_mix(zone=zone)
        return float(mix.gwp) if mix is not None else None
    except ImportError:
        return None


def write_json(path, value):
    """Atomic publication across host/container filesystem views."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def set_phase(name, episode_id=None):
    """Wait for the host's boundary counter reading; no-op outside tracked runs."""
    directory = os.environ.get("COORDINATION_MEASUREMENTS")
    if not directory:
        return
    root = Path(directory)
    request = {"id": uuid.uuid4().hex, "phase": name, "episode_id": episode_id}
    write_json(root / "phase-request.json", request)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if json.loads((root / "phase-ack.json").read_text())["id"] == request["id"]:
                return
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    raise TimeoutError("Host measurement process did not acknowledge phase boundary")


def phase_result(start, end, label):
    """Counter resets/unavailable counters produce missing energy, never zero."""
    seconds = end["monotonic_seconds"] - start["monotonic_seconds"]
    a, b = start.get("gpu_energy_mj"), end.get("gpu_energy_mj")
    energy = (b - a) / 3.6e9 if a is not None and b is not None and b > a else None
    return {
        **label,
        "start_utc": start["utc"],
        "end_utc": end["utc"],
        "start_monotonic_seconds": start["monotonic_seconds"],
        "end_monotonic_seconds": end["monotonic_seconds"],
        "duration_seconds": seconds,
        "gpu_energy_kwh": energy,
        "gpu_counter_start_mj": a,
        "gpu_counter_end_mj": b,
        "time_weighted_gpu_power_w": energy * 3.6e6 / seconds if energy is not None and seconds > 0 else None,
        "gpu_operational_emissions_kg_co2eq_estimate": energy * label["grid_kg_co2eq_per_kwh"]
        if energy is not None and label.get("grid_kg_co2eq_per_kwh") is not None
        else None,
        "cpu_energy_kwh": None,
        "ram_energy_kwh": None,
        "scope": "allocated_gpu_board_counter_delta",
        "energy_status": "measured" if energy is not None else "unavailable_reset_or_no_increment",
    }


def monitor(root, command, allow_cpu=False):
    """Launch one child tree, sample every second, preserve failures and partial phases."""
    import psutil
    import pynvml

    handle = None
    gpu_error = None
    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() != 1:
            raise RuntimeError("Phase tracking requires exactly one visible allocated GPU")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(gpu_uuid, bytes):
            gpu_uuid = gpu_uuid.decode()
        if (root / "gpu-uuids.txt").read_text().strip() != gpu_uuid:
            raise RuntimeError("Allocated GPU and NVML sampler selected different GPUs")
    except pynvml.NVMLError as error:
        if not allow_cpu:
            raise
        gpu_error = str(error)
    packages = {}
    for name in ("ecologits", "nvidia-ml-py", "psutil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    drivers = {}
    for name in ("nvmlSystemGetDriverVersion", "nvmlSystemGetNVMLVersion"):
        try:
            value = getattr(pynvml, name)()
            drivers[name] = value.decode() if isinstance(value, bytes) else value
        except (pynvml.NVMLError, AttributeError):
            drivers[name] = None
    zone = os.environ.get("COORDINATION_ELECTRICITY_ZONE", "NLD")
    factor = grid_factor(zone)
    write_json(
        root / "measurement-environment.json",
        {
            "packages": packages,
            "sample_interval_seconds": 1,
            "electricity_mix_zone": zone,
            "grid_kg_co2eq_per_kwh": factor,
            "driver_versions": drivers,
            "gpu_scope": "single_visible_allocated_board",
            "cpu_ram_utilisation_scope": "host",
            "process_memory_scope": "child_tree_live_processes",
        },
    )
    env = {**os.environ, "COORDINATION_MEASUREMENTS": str(root.resolve())}
    # Apptainer explicitly preserves the phase channel even with a clean environment.
    env["APPTAINERENV_COORDINATION_MEASUREMENTS"] = env["COORDINATION_MEASUREMENTS"]
    child = subprocess.Popen(command, env=env, start_new_session=True)
    process = psutil.Process(child.pid)
    received_signal = None

    def stop(signum, frame):
        nonlocal received_signal
        received_signal = signum
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}

    def sample():
        row = {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "monotonic_seconds": time.monotonic(),
            "gpu_uuid": gpu_uuid if handle is not None else None,
            "gpu_energy_mj": None,
            "gpu_power_w": None,
            "gpu_used_bytes": None,
            "gpu_utilisation_percent": None,
            "gpu_error": gpu_error,
            "gpu_temperature_c": None,
            "gpu_sm_clock_mhz": None,
            "gpu_power_limit_w": None,
        }
        if handle is not None:
            for key, read in {
                "gpu_energy_mj": lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle),
                "gpu_power_w": lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                "gpu_used_bytes": lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                "gpu_utilisation_percent": lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "gpu_temperature_c": lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                "gpu_sm_clock_mhz": lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                "gpu_power_limit_w": lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000,
            }.items():
                try:
                    before = time.monotonic()
                    row[key] = read()
                    if key == "gpu_energy_mj":
                        row["monotonic_seconds"] = (before + time.monotonic()) / 2
                except (pynvml.NVMLError, AttributeError) as error:
                    row[key + "_error"] = str(error)
        # Read host/process statistics after the counter timestamp so slow /proc
        # queries cannot shift the energy/time boundary used for phase watts.
        row.update(host_cpu_percent=psutil.cpu_percent(), host_ram_used_bytes=psutil.virtual_memory().used)
        rss, uss, count = 0, 0, 0
        try:
            descendants = [process, *process.children(recursive=True)]
        except psutil.Error:
            descendants = []
        for descendant in descendants:
            try:
                memory = descendant.memory_full_info()
                rss += memory.rss  # shared pages may be counted more than once
                uss += memory.uss
                count += 1
            except psutil.Error:
                continue
        row.update(
            child_tree_rss_sum_bytes=rss if count else None,
            child_tree_uss_sum_bytes=uss if count else None,
            sampled_processes=count,
        )
        return row

    label = {"phase": "model_startup", "episode_id": None, "grid_kg_co2eq_per_kwh": factor}
    seen, phases, peaks, failure = None, [], {}, None
    started = sample()
    boundary, latest = started, started
    try:
        with (root / "power-samples.jsonl").open("x") as samples, (root / "phases.jsonl").open("x") as phase_stream:
            next_sample = 0
            while True:
                request_path = root / "phase-request.json"
                request = json.loads(request_path.read_text()) if request_path.exists() else None
                changed = request and request["id"] != seen
                if changed or time.monotonic() >= next_sample or child.poll() is not None:
                    latest = sample()
                    samples.write(json.dumps({**label, **latest}) + "\n")
                    samples.flush()
                    for key in ("gpu_power_w", "gpu_used_bytes", "child_tree_uss_sum_bytes", "child_tree_rss_sum_bytes"):
                        if latest[key] is not None:
                            peaks[key] = max(peaks.get(key, 0), latest[key])
                    next_sample = time.monotonic() + 1
                if changed:
                    result = phase_result(boundary, latest, label)
                    phases.append(result)
                    phase_stream.write(json.dumps(result) + "\n")
                    phase_stream.flush()
                    boundary, label, seen = latest, {**request, "grid_kg_co2eq_per_kwh": factor}, request["id"]
                    write_json(root / "phase-ack.json", {"id": seen})
                if child.poll() is not None:
                    result = phase_result(boundary, latest, label)
                    phases.append(result)
                    phase_stream.write(json.dumps(result) + "\n")
                    break
                if received_signal:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                time.sleep(0.02)
    except BaseException as error:
        failure = repr(error)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        raise
    finally:
        code = child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write_json(
            root / "measurement-summary.json",
            {
                "status": "complete" if code == 0 and failure is None else "partial_or_failed",
                "child_exit_code": code,
                "sampler_error": failure,
                "sample_interval_seconds": 1,
                "sampled_maxima": peaks,
                "phases": phases,
                "whole_sampler_window": phase_result(
                    started, latest, {"phase": "whole_sampler_window", "grid_kg_co2eq_per_kwh": factor}
                ),
                "cpu_ram_energy_source": "unavailable",
                "grid_factor_source": "EcoLogits electricity_mix_repository",
                "electricity_mix_zone": zone,
                "grid_kg_co2eq_per_kwh": factor,
                "do_not_add_nvml_energy_to_ecologits_estimates": True,
            },
        )
        if handle is not None:
            pynvml.nvmlShutdown()
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    from codecarbon import OfflineEmissionsTracker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not os.environ.get("SLURM_JOB_ID"):
        parser.error("Provide a child command on an allocated Slurm GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    uuid_text = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not uuid_text.startswith("GPU-") or "\n" in uuid_text:
        raise ValueError("Exactly one allocated GPU must be visible")
    (args.output / "gpu-uuids.txt").write_text(uuid_text + "\n")
    (args.output / "slurm-job-id.txt").write_text(os.environ["SLURM_JOB_ID"] + "\n")
    tracker = OfflineEmissionsTracker(
        country_iso_code="NLD",
        gpu_ids="0",
        tracking_mode="process",
        output_dir=str(args.output),
        output_file="emissions.csv",
        measure_power_secs=5,
        pue=1.0,
        save_to_api=False,
        log_level="warning",
    )
    tracker.start()
    try:
        code = monitor(args.output, command)
    finally:
        tracker.stop()
    run_root = os.environ.get("COORDINATION_RUN_ROOT")
    if run_root:
        estimate_saved_requests(Path(run_root), args.output, job_id=os.environ.get("SLURM_JOB_ID"))
    raise SystemExit(code)
