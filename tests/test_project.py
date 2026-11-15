"""CPU checks for the experiment glue; these never call a model or spend SBU."""

import ast
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

import language
import run
from controller import Controller


def sample_configuration():
    """Public checks use a synthetic balance, never a researcher's account data."""
    return {**run.configuration(), "balance_sbu": 20000, "reserve_sbu": 2000}


def test_localization_preserves_expressions_and_protocol_keys():
    source = '''
def prompt(player):
    """Documentation is not an actor instruction."""
    return f"Choose an action for {player}: {{'action_plan': []}}."
def parse(response):
    return response['action_plan']
'''
    texts = language.inventory(source)
    assert len(texts) == 1
    key, text = next(iter(texts.items()))
    translated = text.replace("Choose an action for", "Kies een actie voor")
    code = language.localize(source, {key: translated})
    namespace = {}
    exec(compile(code, "<test>", "exec"), namespace)  # noqa: S102 -- This fixture checks the generated prompt's actual behaviour.
    assert namespace["prompt"]("player_0") == "Kies een actie voor player_0: {'action_plan': []}."
    assert namespace["parse"]({"action_plan": [1]}) == [1]
    assert ast.dump(language.localize(source, texts)) == ast.dump(ast.parse(source))


def test_translation_cannot_drop_slots_or_change_numbers():
    with pytest.raises(ValueError, match="placeholder"):
        language.validate("Give player ⟦0⟧ 20 points.", "Geef 20 punten.")
    with pytest.raises(ValueError, match="number"):
        language.validate("Give player ⟦0⟧ 20 points.", "Geef speler ⟦0⟧ 2 punten.")


def test_inventory_keeps_parser_constants_and_paths():
    source = """
def parse(text):
    if text == 'not found':
        return {}
    return text.split('action plan: ')
def prompt():
    return 'Choose your next action.'
"""
    assert list(language.inventory(source).values()) == ["Choose your next action."]


def test_localization_cannot_translate_fences_actions_or_control_tags():
    source = """
def parse(response):
    start_marker = "```python\\n"
    return response.find(start_marker)
def action(src, dst):
    return f"move_to({src}, {dst})"
def status(where):
    tag = f"standing around at {where}"
    return 'standing around at' in tag
def prompt():
    return "Choose your next action."
"""
    assert list(language.inventory(source).values()) == ["Choose your next action."]


def test_all_upstream_templates_round_trip_in_english():
    texts = language.source_texts()
    assert len(texts) > 30
    for path in language.sources():
        source = path.read_text()
        assert ast.dump(language.localize(source, texts)) == ast.dump(ast.parse(source))


def test_allocation_is_balanced_and_rejects_overspending():
    config = sample_configuration()
    plan = run.allocation(config, "collection", episode_minutes=50)
    assert len(plan["episodes"]) == 96
    assert len(plan["tasks"]) == 8
    assert len({e["environment_seed"] for e in plan["episodes"]}) == 96
    for task in plan["tasks"]:
        assert {(plan["episodes"][i]["language"], plan["episodes"][i]["thinking"]) for i in task} == {
            (lang, mode) for lang in config["languages"] for mode in (False, True)
        }
    assert plan["maximum_gpu_sbu"] == 15744
    with pytest.raises(ValueError, match="budget"):
        run.allocation(config, "collection", episode_minutes=70)
    with pytest.raises(ValueError):
        run.allocation(config, "collection", episode_minutes=float("nan"))


def test_pilot_covers_games_modes_and_languages_in_one_job():
    plan = run.allocation(sample_configuration(), "pilot", episode_minutes=50)
    assert len(plan["tasks"]) == 1
    assert len(plan["episodes"]) == 18
    assert plan["maximum_gpu_sbu"] == 384
    assert len({(e["game"], e["thinking"]) for e in plan["episodes"] if e["language"] == "en"}) == 8
    assert {e["language"] for e in plan["episodes"]} == {"en", "nl", "fr", "es", "ar", "zh"}


def test_controller_logs_native_usage_and_retains_failed_requests(tmp_path):
    controller = Controller("http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, False, 71)

    async def reply(payload):
        return {
            "model": "Qwen/test",
            "choices": [{"message": {"content": "{'action_plan': []}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }

    controller.request = reply
    assert asyncio.run(controller.async_batch_prompt("rules", ["act"])) == ["{'action_plan': []}"]
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    assert row["completion_tokens"] == 4
    assert row["prompt_tokens"] == 12
    assert row["messages"][0]["content"] == "rules"
    assert row["status"] == "complete"

    async def broken(payload):
        raise TimeoutError("server stopped")

    controller.request = broken
    with pytest.raises(TimeoutError):
        asyncio.run(controller.async_batch_prompt("rules", ["act"]))
    assert json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[-1])["status"] == "failed"


def test_native_metrics_are_evaluator_only(tmp_path):
    from hm import Scenario

    observable = lambda: SimpleNamespace(subscribe=lambda callback: SimpleNamespace(dispose=lambda: None))
    factory = SimpleNamespace(num_focal_players=lambda: 1)
    native = SimpleNamespace(
        observables=lambda: SimpleNamespace(
            substrate=SimpleNamespace(action=observable(), timestep=observable(), events=observable())
        )
    )
    scenario = Scenario(native, factory, tmp_path, 2, "prisoners_dilemma_in_the_matrix__repeated_0")
    scenario.observe(SimpleNamespace(reward=(3.0, -1.0), observation=({"RGB": 0}, {"RGB": 0})))
    assert scenario.returns == [3.0, -1.0]
    assert scenario.summary()["collective_return"] == 2.0


def test_pd_cooperation_uses_executed_inventory_and_handles_role_order(tmp_path):
    from hm import Scenario

    observable = lambda: SimpleNamespace(subscribe=lambda callback: SimpleNamespace(dispose=lambda: None))
    native = SimpleNamespace(
        observables=lambda: SimpleNamespace(
            substrate=SimpleNamespace(action=observable(), timestep=observable(), events=observable())
        )
    )
    scenario = Scenario(
        native, SimpleNamespace(num_focal_players=lambda: 1), tmp_path, 2, "prisoners_dilemma_in_the_matrix__repeated_0"
    )
    assert scenario.summary()["pd_cooperation_share"] is None
    scenario.event(
        ("interaction", [b"dict", b"row_player_idx", 1, b"col_player_idx", 2, b"row_inventory", [3, 1], b"col_inventory", [1, 3]])
    )
    scenario.event(
        ("interaction", [b"dict", b"row_player_idx", 2, b"col_player_idx", 1, b"row_inventory", [1, 3], b"col_inventory", [1, 1]])
    )
    assert scenario.summary()["pd_interactions"] == 2
    assert scenario.summary()["pd_cooperation_share"] == pytest.approx(0.625)
    assert scenario.summary()["pd_partner_cooperation_share"] == pytest.approx(0.25)


def test_missing_episodes_remain_in_saved_data_report(tmp_path):
    import csv

    data = run.allocation(sample_configuration(), "pilot", episode_minutes=50)
    (tmp_path / "plan.json").write_text(json.dumps(data))
    result = run.report(tmp_path)
    assert result["episodes"] == 18
    assert result["complete"] == 0
    with (tmp_path / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["status"] == "not_started" and row["gpu_energy_kwh"] == "" for row in rows)


def test_gpu_energy_conversion_and_unsupported_request_profile():
    from measure import estimate_request, phase_result

    start = {"monotonic_seconds": 0.0, "utc": "start", "gpu_energy_mj": 1000}
    end = {"monotonic_seconds": 10.0, "utc": "end", "gpu_energy_mj": 1001000}
    result = phase_result(start, end, {"grid_kg_co2eq_per_kwh": 0.1})
    assert result["gpu_energy_kwh"] == pytest.approx(1000 / 3.6e6)
    assert result["time_weighted_gpu_power_w"] == pytest.approx(100)
    assert phase_result(end, start, {})["gpu_energy_kwh"] is None
    assert estimate_request({"completion_tokens": 10})["status"] == "unsupported_model_or_deployment"


def test_nan_budget_is_rejected():
    config = sample_configuration()
    config["balance_sbu"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run.allocation(config, "collection", episode_minutes=50)


def test_pd_runner_respects_environment_termination(tmp_path):
    folder = tmp_path / "environments"
    folder.mkdir()
    path = folder / "prisoners_dilemma_in_the_matrix__repeated.py"
    path.write_text(
        "def run_episode(env):\n    done = False\n    step = 0\n    while step < 1200:\n        step += 1\n        done = env.step(step)\n    return step\n"
    )
    module = language.load_module("pd_stop_fixture", path)
    assert module.run_episode(SimpleNamespace(step=lambda step: step >= 3)) == 3


@pytest.mark.parametrize("exit_code", [0, 7])
def test_measurement_preserves_child_failure_without_a_gpu(tmp_path, monkeypatch, exit_code):
    pytest.importorskip("psutil")
    pynvml = pytest.importorskip("pynvml")
    import measure

    def unavailable():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND)

    monkeypatch.setattr(pynvml, "nvmlInit", unavailable)
    assert measure.monitor(tmp_path, [sys.executable, "-c", f"raise SystemExit({exit_code})"], allow_cpu=True) == exit_code
    result = json.loads((tmp_path / "measurement-summary.json").read_text())
    assert result["child_exit_code"] == exit_code
    assert result["whole_sampler_window"]["gpu_energy_kwh"] is None
