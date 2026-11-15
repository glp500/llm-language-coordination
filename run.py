"""Plan a bounded allocation, run pinned Hypothetical Minds, or summarize saved evidence."""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from functools import partial
from pathlib import Path
from unittest.mock import patch

import yaml

import language
from controller import Controller, save

ROOT = Path(__file__).resolve().parent


def configuration():
    """One human-readable study configuration; no KG or historical controller is loaded."""
    config = yaml.safe_load((ROOT / "experiment.yaml").read_text())
    budget = ROOT / ".budget.yaml"
    if budget.exists():
        values = yaml.safe_load(budget.read_text())
        if set(values) != {"balance_sbu", "reserve_sbu"}:
            raise ValueError("Private .budget.yaml must contain balance_sbu and reserve_sbu only")
        config.update(values)
    return config


def allocation(config, stage, episode_minutes):
    """Allocate complete randomized language/mode blocks before any GPU request."""
    if stage not in {"pilot", "collection"} or not math.isfinite(episode_minutes) or episode_minutes <= 0:
        raise ValueError("Choose a valid stage and positive finite episode minutes")
    for field in ("balance_sbu", "sbu_per_gpu_hour", "reserve_sbu", "pilot_minutes", "startup_minutes"):
        if not isinstance(config[field], (int, float)) or not math.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"{field} must be positive and finite")
    if len(config["languages"]) != 6 or set(config["languages"]) != set(language.LANGUAGES):
        raise ValueError("Keep all six declared language conditions")
    if len(config["games"]) != 4 or set(config["games"]) != set(language.GAMES):
        raise ValueError("Keep all four declared HM games")
    if config["thinking"] != [False, True] or not isinstance(config["replicates"], int) or config["replicates"] < 1:
        raise ValueError("Declare both thinking modes and positive independent replication")
    if not isinstance(config["frames"], int) or not 1 <= config["frames"] <= 1500:
        raise ValueError("Use a horizon of 1–1500 native frames")
    rng, episodes, tasks = random.Random(config["seed"]), [], []
    conditions = [(lang, mode) for lang in config["languages"] for mode in config["thinking"]]
    for replicate in range(1 if stage == "pilot" else config["replicates"]):
        games = list(config["games"])
        rng.shuffle(games)
        for game in games:
            cells = [("en", mode) for mode in config["thinking"]] if stage == "pilot" else conditions.copy()
            if stage == "pilot" and game == "cc":
                cells += [(lang, mode) for lang, mode in conditions if lang != "en"]
            rng.shuffle(cells)
            task = []
            for lang, mode in cells:
                index = len(episodes)
                task.append(index)
                episodes.append(
                    {
                        "id": f"episode-{index:04d}",
                        "game": game,
                        "language": lang,
                        "thinking": mode,
                        "scenario": config["scenario"],
                        "replicate": replicate,
                        "frames": (200 if lang == "en" else 100) if stage == "pilot" else config["frames"],
                        "environment_seed": config["seed"] + 100000 + index,
                        "sampling_seed": config["seed"] + 200000 + index,
                    }
                )
            tasks.append(task)
    if stage == "pilot":
        tasks = [[index for task in tasks for index in task]]
        walls = [config["pilot_minutes"]]
    else:
        walls = [config["startup_minutes"] + len(task) * episode_minutes for task in tasks]
    maximum = sum(walls) / 60 * config["sbu_per_gpu_hour"]
    committed = 0 if stage == "pilot" else config["pilot_minutes"] / 60 * config["sbu_per_gpu_hour"]
    if maximum + committed + config["reserve_sbu"] > config["balance_sbu"]:
        raise ValueError("Allocation exceeds the budget after the pilot ceiling and reserve")
    return {
        "study": config["study"],
        "stage": stage,
        "config": config,
        "episodes": episodes,
        "tasks": tasks,
        "task_wall_minutes": walls,
        "episode_minutes": episode_minutes,
        "maximum_gpu_sbu": maximum,
        "pilot_ceiling_accounted_sbu": committed,
        "reserve_sbu": config["reserve_sbu"],
        "runtime_forecast": False,
        "confirmatory": False,
    }


def source_hashes():
    """Freeze execution glue, selected upstream code and configuration independently of Git's index."""
    paths = [
        *ROOT.glob("*.py"),
        ROOT / "experiment.yaml",
        ROOT / "environment.yml",
        ROOT / "requirements.txt",
        *sorted((ROOT / "translations").glob("*.json")),
        *language.sources(),
        language.UPSTREAM / "llm_plan/env/mp_llm_env.py",
        language.UPSTREAM / "llm_plan/agent/action_funcs.py",
        language.UPSTREAM / "utils.py",
        *sorted((language.UPSTREAM / "llm_plan/sprite_labels").rglob("*.png")),
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def plan(output, stage, minutes, pilot=None):
    """Create a new self-contained plan; collection requires a completed pilot with the same sources."""
    config = configuration()
    value = allocation(config, stage, minutes)
    hashes = source_hashes()
    if stage == "collection":
        if pilot is None:
            raise ValueError("Collection requires --pilot pointing to the completed pilot allocation")
        previous = json.loads((pilot / "plan.json").read_text())
        if previous["stage"] != "pilot" or previous["source_hashes"] != hashes:
            raise ValueError("Pilot and collection must use identical execution sources")
        completed = json.loads((pilot / "tasks/0000.json").read_text())
        if completed["status"] != "complete":
            raise ValueError("The bounded end-to-end pilot has not completed")
        for spec in previous["episodes"]:
            result = json.loads((pilot / spec["id"] / "summary.json").read_text())
            if result["status"] != "complete" or result["frames"] < 1:
                raise ValueError("Pilot has missing or failed gameplay")
        measurement = json.loads((pilot / "energy/0000/measurement-summary.json").read_text())
        if (
            measurement["status"] != "complete"
            or measurement["child_exit_code"] != 0
            or not (measurement["whole_sampler_window"].get("gpu_energy_kwh", 0) or 0) > 0
        ):
            raise ValueError("Pilot needs successful measured GPU energy")
        import csv

        with (pilot / "energy/0000/emissions.csv").open() as stream:
            carbon = list(csv.DictReader(stream))
        if len(carbon) != 1 or carbon[0]["codecarbon_version"] != "3.3.2" or float(carbon[0]["gpu_energy"]) <= 0:
            raise ValueError("Pilot needs the pinned CodeCarbon GPU measurement")
        impacts = json.loads((pilot / "energy/0000/impacts-summary.json").read_text())
        if impacts["version"] != "0.11.3" or impacts["requests"] < 1:
            raise ValueError("Pilot needs EcoLogits request accounting, including unavailable statuses")
        for lang in config["languages"]:
            corpus = json.loads((pilot / "translations" / f"{lang}.json").read_text())
            for key, text in language.source_texts().items():
                language.validate(text, corpus["texts"][key])
    output.mkdir(parents=True, exist_ok=False)
    value["source_hashes"] = hashes
    value["hm_commit"] = subprocess.check_output(["git", "-C", str(language.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if value["hm_commit"] != config["hm_revision"]:
        raise ValueError("HM checkout differs from the pinned revision")
    for name in hashes:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    if stage == "collection":
        shutil.copytree(pilot / "translations", output / "translations")
    else:
        shutil.copytree(ROOT / "translations", output / "translations")
    save(output / "plan.json", value)
    return {
        "episodes": len(value["episodes"]),
        "jobs": len(value["tasks"]),
        "maximum_gpu_sbu": value["maximum_gpu_sbu"],
        "task_wall_minutes": value["task_wall_minutes"],
    }


def episode(root, index):
    """Run HM's original agent and loop in a fresh process, with bounded frames and full evidence."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("COORDINATION_MEASUREMENTS"):
        raise ValueError("Run episodes through the measured allocation, not as an untracked command")
    import numpy as np
    import tensorflow as tf
    from meltingpot.utils.substrates import builder

    import hm
    from measure import set_phase

    plan_data = json.loads((root / "plan.json").read_text())
    spec, config = plan_data["episodes"][index], plan_data["config"]
    output = root / spec["id"]
    output.mkdir(exist_ok=False)
    manifest = {
        **spec,
        "status": "partial",
        "model": config["model"],
        "hm_commit": config["hm_revision"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "monetary_inference_cost": None,
    }
    save(output / "manifest.json", manifest)
    env, actor = None, None
    try:
        tf.config.set_visible_devices([], "GPU")
        random.seed(spec["environment_seed"])
        np.random.seed(spec["environment_seed"])
        tf.random.set_seed(spec["environment_seed"])
        corpus = json.loads((root / "translations" / f"{spec['language']}.json").read_text())
        manifest["translation_reviewed"] = corpus["reviewed"]
        manifest["corpus_sha256"] = hashlib.sha256((root / "translations" / f"{spec['language']}.json").read_bytes()).hexdigest()
        sys.path.insert(0, str(language.UPSTREAM))
        game, agent_file = language.GAMES[spec["game"]]
        agent_module = language.load_module(
            "hm_agent", language.UPSTREAM / "llm_plan/agent" / game / f"{agent_file}.py", corpus["texts"]
        )
        runner = language.load_module("hm_episode", language.UPSTREAM / "environments" / f"{game}.py", corpus["texts"])
        perception = language.load_module("hm_perception", language.UPSTREAM / "llm_plan/env/mp_llm_env.py", perception=True)
        controller = Controller(
            config["server_url"],
            config["model"],
            output,
            spec["thinking"],
            spec["sampling_seed"],
            config["request_tokens"],
            config["request_limit"],
        )
        actor = agent_module.DecentralizedAgent({"agent_id": "player_0", "self_improve": True}, controller)
        actor.system_message += "\n" + corpus["response_instruction"]
        actor.agent_type, actor.llm_type = "hm", "gpt4"  # Select HM's generic parser, independent of server model name.
        hm.OUTPUT, hm.FRAME_LIMIT = output, spec["frames"]
        sprites = output / "sprites" / game
        shutil.copytree(language.UPSTREAM / "llm_plan/sprite_labels" / game, sprites)
        (sprites.parent / "unlabeled_patches").mkdir()
        set_phase("environment_setup", spec["id"])
        with patch.object(builder, "builder", partial(builder.builder, env_seed=spec["environment_seed"])):
            env = perception.MeltingPotLLMEnv(game, str(sprites), spec["scenario"])
        (output / "results").mkdir()
        os.chdir(output)
        set_phase("gameplay", spec["id"])
        with (
            (output / "console.log").open("w") as console,
            contextlib.redirect_stdout(console),
            contextlib.redirect_stderr(console),
        ):
            frames = asyncio.run(runner.run_episode(env, actor))
        manifest.update(
            status="complete",
            focal_return=env._env.returns[0],
            frame_folder=frames,
            requests=controller.calls,
            **env._env.summary(),
        )
    except BaseException as error:
        manifest["error"] = repr(error)
        raise
    finally:
        if env is not None:
            manifest.update(env._env.summary())
            env._env.close()
        if actor is not None:
            manifest["hypotheses"] = {key: value for key, value in actor.__dict__.items() if "hypothes" in key}
        save(output / "summary.json", manifest)
        save(output / "manifest.json", manifest)


def batch(root, index):
    """One resident Qwen server per allocation block; failures stop the block without automatic reruns."""
    from measure import set_phase

    value = json.loads((root / "plan.json").read_text())
    config = value["config"]
    if value["source_hashes"] != source_hashes():
        raise ValueError("Execution sources changed; make a fresh allocation")
    if not 0 <= index < len(value["tasks"]):
        raise ValueError("Task index outside allocation")
    installed = importlib.metadata.distribution("dm-meltingpot").read_text("direct_url.json")
    if not installed or json.loads(installed).get("vcs_info", {}).get("commit_id") != config["meltingpot_revision"]:
        raise ValueError("Install the pinned HM Melting Pot fork into the new Conda environment")
    # This is checked before starting a model process or issuing a request.
    allocated_seconds = int(os.environ.get("COORDINATION_ALLOCATED_SECONDS", "0"))
    if not os.environ.get("SLURM_JOB_ID") or allocated_seconds != int(value["task_wall_minutes"][index] * 60):
        raise ValueError("Use job.sbatch with exactly the frozen task wall time")
    (root / "tasks").mkdir(exist_ok=True)
    path = root / "tasks" / f"{index:04d}.json"
    if path.exists():
        raise ValueError("Task already attempted; preserve its outputs and allocate a new run")
    state = {
        "status": "partial",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "completed_episodes": [],
        "package_versions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
    }
    save(path, state)
    server = None
    try:
        # A pre-existing server would be outside the measured child tree and frozen startup policy.
        try:
            urllib.request.urlopen(config["server_url"] + "/models", timeout=1).close()
        except OSError:
            pass
        else:
            raise RuntimeError("Port is occupied; stop the existing model server before this allocation")
        with (root / "tasks" / f"server-{index:04d}.log").open("w") as log:
            server = subprocess.Popen(config["server_command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + config["startup_minutes"] * 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Model server exited; inspect the saved server log")
                try:
                    with urllib.request.urlopen(config["server_url"] + "/models", timeout=2) as response:
                        models = json.load(response)
                    if config["model"] not in {m["id"] for m in models["data"]}:
                        raise ValueError("Server exposed a different model")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Model startup exceeded its frozen allowance")
                    time.sleep(1)
            if value["stage"] == "pilot":
                set_phase("translation_preparation", "preparation")
                output = root / "preparation"
                output.mkdir(exist_ok=False)
                translator = Controller(
                    config["server_url"], config["model"], output, False, config["seed"], config["request_tokens"], 100000
                )
                asyncio.run(language.prepare(root / "translations", translator))
            for row in value["tasks"][index]:
                started = time.monotonic()
                subprocess.run(
                    [
                        "apptainer",
                        "exec",
                        str(ROOT / "harvest.sif"),
                        sys.executable,
                        str(ROOT / "run.py"),
                        "episode",
                        str(root),
                        "--index",
                        str(row),
                    ],
                    cwd=ROOT,
                    check=True,
                    timeout=value["episode_minutes"] * 60,
                )
                state["completed_episodes"].append({"index": row, "seconds": time.monotonic() - started})
                save(path, state)
                set_phase("between_episodes", value["episodes"][row]["id"])
            state["status"] = "complete"
    except BaseException as error:
        state["error"] = repr(error)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        save(path, state)


def report(root):
    """Read saved outputs only; failures and missing measurements remain explicit."""
    import csv

    rows = []
    allocation_data = json.loads((root / "plan.json").read_text())
    phases = [
        phase
        for path in sorted(root.glob("energy/*/measurement-summary.json"))
        for phase in json.loads(path.read_text())["phases"]
    ]
    for spec in allocation_data["episodes"]:
        folder = root / spec["id"]
        path, manifest = folder / "summary.json", folder / "manifest.json"
        value = (
            json.loads(path.read_text())
            if path.exists()
            else json.loads(manifest.read_text())
            if manifest.exists()
            else {**spec, "status": "not_started"}
        )
        requests = folder / "requests.jsonl"
        calls = [json.loads(line) for line in requests.read_text().splitlines()] if requests.exists() else []
        row = {
            key: value.get(key)
            for key in (
                "id",
                "game",
                "language",
                "thinking",
                "replicate",
                "status",
                "frames",
                "focal_return",
                "collective_return",
                "per_capita_return",
                "pd_interactions",
                "pd_cooperation_share",
                "pd_partner_cooperation_share",
                "translation_reviewed",
            )
        }
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens_reported"):
            counts = [call.get(field) for call in calls]
            row[field] = sum(counts) if counts and all(count is not None for count in counts) else None
        row["requests"], row["failed_requests"] = len(calls), sum(call["status"] != "complete" for call in calls)
        row["request_wall_seconds_sum"] = sum(
            call["backend_batch_end_monotonic"] - call["backend_batch_start_monotonic"] for call in calls
        )
        measured = [
            phase
            for phase in phases
            if phase.get("episode_id") == spec["id"] and phase["phase"] in {"environment_setup", "gameplay"}
        ]
        energy = [phase["gpu_energy_kwh"] for phase in measured]
        row["gpu_energy_kwh"] = sum(energy) if energy and all(e is not None for e in energy) else None
        row["measured_window_seconds"] = sum(phase["duration_seconds"] for phase in measured) if measured else None
        row["mean_gpu_power_w"] = (
            row["gpu_energy_kwh"] * 3.6e6 / row["measured_window_seconds"]
            if (row["gpu_energy_kwh"] is not None and row["measured_window_seconds"])
            else None
        )
        rows.append(row)
    if rows:
        with (root / "episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "episodes": len(rows),
        "complete": sum(row["status"] == "complete" for row in rows),
        "output": str(root / "episodes.csv"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "batch", "episode", "report"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=("pilot", "collection"), default="pilot")
    parser.add_argument("--episode-minutes", type=float, default=50)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "plan":
        result = plan(root, args.stage, args.episode_minutes, args.pilot.resolve() if args.pilot else None)
    elif args.command == "batch":
        result = batch(root, args.index)
    elif args.command == "episode":
        result = episode(root, args.index)
    else:
        result = report(root)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
