"""CPU checks for the experiment glue; these never call a model or spend SBU."""

import ast
import asyncio
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

import language
import run
from controller import Controller


def sample_configuration():
    """Public checks use a synthetic balance, never a researcher's account data."""
    return {**run.configuration(), "balance_sbu": 20000, "reserve_sbu": 2000}


def test_batch_uses_project_python_when_module_python_is_first(tmp_path, monkeypatch):
    """Exercise the real batch shell without Slurm/GPU; reactivation may leave PATH stale."""
    project = tmp_path / "project with spaces"
    runtime = project / ".conda/runtime/bin/python"
    runtime.parent.mkdir(parents=True)
    module = tmp_path / "module"
    profile = module / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text('conda() { export CONDA_PREFIX="$2"; }\n')
    commands = module / "bin"
    commands.mkdir()
    for name, body in {
        "module": "exit 0\n",
        "conda": f"printf '%s\\n' {shlex.quote(str(module))}\n",
        "squeue": "printf '02:00:00\\n'\n",
        "python": "exit 91\n",  # Base Python must never execute the measurement or batch.
    }.items():
        path = commands / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    captured = tmp_path / "arguments.txt"
    runtime.write_text(
        '#!/bin/sh\nif [ "$1" = "-c" ]; then\n'
        f"exec {shlex.quote(sys.executable)} \"$@\"\nfi\n"
        f"printf '%s\\n' \"$COORDINATION_ALLOCATED_SECONDS\" \"$@\" > {shlex.quote(str(captured))}\n"
    )
    runtime.chmod(0o755)
    monkeypatch.setenv("PATH", f"{commands}:/usr/bin:/bin")
    monkeypatch.setenv("SLURM_SUBMIT_DIR", str(project))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    output = project / "outputs/pilot-01"
    result = subprocess.run(
        ["bash", str(run.ROOT / "job.sbatch"), str(output), "0"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    arguments = captured.read_text().splitlines()
    assert arguments[:3] == ["7200", "measure.py", str(output / "energy/0000")]
    assert arguments[-6:] == [str(runtime), "run.py", "batch", str(output), "--index", "0"]


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


def test_payoff_correction_preserves_prose_and_uses_both_inventory_sums():
    formula = "r_t = transpose(your_inventory) * A_payoff * opponent_inventory"
    original = f"Regels blijven gelijk.\n{formula}\nKeep 'action_plan' unchanged."
    corrected = language.correct_payoff(original)
    assert corrected == (
        "Regels blijven gelijk.\n"
        "r_t = (transpose(your_inventory) * A_payoff * opponent_inventory) / "
        "(sum(your_inventory) * sum(opponent_inventory))\nKeep 'action_plan' unchanged."
    )
    assert original.count(formula) == 1


@pytest.mark.parametrize("text", ["missing formula", "r_t = transpose(your_inventory) * A_payoff * opponent_inventory\n" * 2])
def test_payoff_correction_rejects_missing_or_ambiguous_source(text):
    with pytest.raises(ValueError, match="payoff formula"):
        language.correct_payoff(text)


def test_translation_cannot_drop_slots_or_change_numbers():
    with pytest.raises(ValueError, match="placeholder"):
        language.validate("Give player ⟦0⟧ 20 points.", "Geef 20 punten.")
    with pytest.raises(ValueError, match="number"):
        language.validate("Give player ⟦0⟧ 20 points.", "Geef speler ⟦0⟧ 2 punten.")


@pytest.mark.parametrize(
    "translated",
    [
        "الأجزاء1، 2، و3؛ player_0 يحصل على2.80.",
        "部分1、2和3；player_0获得2.80。",
        "Delen1, 2en3; player_0 krijgt2.80.",
    ],
)
def test_number_validation_is_independent_of_surrounding_language(translated):
    language.validate("Parts 1, 2, and 3; player_0 gets 2.80.", translated)


def test_number_validation_checks_both_grid_dimensions():
    with pytest.raises(ValueError, match="number"):
        language.validate("A 23x15 grid.", "Een 23x16 raster.")


@pytest.mark.parametrize(
    "translated",
    ["Magnitude 3 to 5; reward -5.", "幅度3到5；奖励-5。", "المقدار3 إلى5؛ المكافأة-5."],
)
def test_number_validation_distinguishes_range_separator_from_negative_sign(translated):
    language.validate("Magnitude 3-5; reward -5.", translated)


@pytest.mark.parametrize(
    ("source", "translated"),
    [("Reward -5.", "Reward 5."), ("Reward -10.", "Reward 10."), ("Range 3-5.", "Range 3 to -5."),
     ("Rewards (5, -5).", "Rewards (5, 5).")],
)
def test_number_validation_still_rejects_changed_negative_payoffs(source, translated):
    with pytest.raises(ValueError, match="number"):
        language.validate(source, translated)


def test_translation_protects_ordinals_protocol_and_existing_expression_slots():
    source = "In the 2nd part, give player_0 ⟦0⟧ 2.80 points using 'cooperate/green' and interact((3, 1))."
    masked, literals = language.protect(source)
    assert language.SLOTS.findall(masked)[0] != "0"  # New slots must not collide with HM expressions.
    assert "⟦0⟧" in masked
    assert all(token not in masked for token in ("2nd", "2.80", "player_0", "cooperate/green", "interact"))
    translated = masked.replace("In the", "In het").replace("part", "deel")
    language.validate(masked, translated)
    restored = language.SLOTS.sub(lambda match: literals.get(match[0], match[0]), translated)
    assert restored == source.replace("In the", "In het").replace("part", "deel")
    language.validate(source, restored)


def test_protected_translation_round_trips_every_upstream_template():
    for source in language.source_texts().values():
        masked, literals = language.protect(source)
        assert not any(character.isdigit() for character in language.SLOTS.sub("", masked))
        assert language.SLOTS.sub(lambda match, saved=literals: saved.get(match[0], match[0]), masked) == source
        language.validate(masked, masked)
        if literals:
            marker = next(iter(literals))
            with pytest.raises(ValueError, match="placeholder"):
                language.validate(masked, masked.replace(marker, "", 1))


@pytest.mark.parametrize("dimensions", ["23x15", "5x5", "5X5", "5×5"])
def test_grid_dimensions_are_protected_as_one_literal(dimensions):
    source = f"You observe a {dimensions} grid around ⟦0⟧."
    masked, literals = language.protect(source)
    assert dimensions in literals.values()
    assert not any(character.isdigit() for character in language.SLOTS.sub("", masked))
    translated = masked.replace("You observe a", "تشاهد شبكة").replace("grid around", "حول")
    language.validate(masked, translated)
    restored = language.SLOTS.sub(lambda match: literals.get(match[0], match[0]), translated)
    language.validate(source, restored)
    assert dimensions in restored


@pytest.mark.parametrize("rejected", ['{"text": "Kies \'actie_plan\' voor ⟦0⟧: 2 acties."}', "{}", "invalid JSON"])
def test_translation_retries_invalid_output_and_preserves_requests(tmp_path, monkeypatch, rejected):
    source = "Choose 'action_plan' for ⟦0⟧: 2 actions."
    translated = "Kies 'action_plan' voor ⟦0⟧: 2 acties."
    monkeypatch.setattr(language, "source_texts", lambda: {"template": source})
    monkeypatch.setattr(language, "LANGUAGES", {"nl": "Dutch"})
    controller = Controller("http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, False, 71)
    answers = iter([rejected])

    async def reply(payload):
        answer = next(answers, None)
        if answer is None:
            text = payload["messages"][1]["content"].replace("Choose", "Kies").replace("for", "voor").replace("actions", "acties")
            answer = json.dumps({"text": text})
        return {"choices": [{"message": {"content": answer}}]}

    controller.request = reply
    directory = tmp_path / "translations"
    asyncio.run(language.prepare(directory, controller))
    corpus = json.loads((directory / "nl.json").read_text())
    assert corpus["texts"] == {"template": translated}
    assert corpus["reviewed"] is False
    requests = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(requests) == 2
    assert requests[0]["response"]["choices"][0]["message"]["content"] == rejected
    assert "Previous translation rejected:" in requests[1]["messages"][0]["content"]
    assert "action_plan" not in requests[1]["messages"][1]["content"]
    rejected_record = json.loads((tmp_path / "translation-validation.jsonl").read_text())
    assert rejected_record["request_call"] == 0
    assert rejected_record["attempt"] == 1
    asyncio.run(language.prepare(directory, controller))
    assert controller.calls == 2  # Frozen valid translations must not be regenerated.


def test_translation_stops_after_three_invalid_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(language, "source_texts", lambda: {"template": "Use 'action_plan'."})
    monkeypatch.setattr(language, "LANGUAGES", {"nl": "Dutch"})
    controller = Controller("http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, False, 71)

    async def reply(payload):
        return {"choices": [{"message": {"content": json.dumps({"text": "Gebruik 'actie_plan'."})}}]}

    controller.request = reply
    with pytest.raises(ValueError, match="after 3 attempts"):
        asyncio.run(language.prepare(tmp_path / "translations", controller))
    assert controller.calls == 3
    assert not (tmp_path / "translations/nl.json").exists()
    failures = [json.loads(line) for line in (tmp_path / "translation-validation.jsonl").read_text().splitlines()]
    assert [row["request_call"] for row in failures] == [0, 1, 2]


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
    assert plan["maximum_gpu_sbu"] == 768
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


@pytest.mark.parametrize("thinking", [False, True])
def test_configured_qwen_sampling_reaches_hm_calls_and_request_logs(tmp_path, thinking):
    """HM's batch interface must not replace the model profile with legacy temperature 0.2."""
    sampling = sample_configuration()["sampling"]
    controller = Controller(
        "http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, thinking, 71,
        max_tokens=8000, thinking_token_budget=3000, sampling=sampling,
    )

    async def reply(payload):
        assert {key: payload[key] for key in sampling} == sampling
        assert payload["temperature"] == 1.0 and payload["presence_penalty"] == 1.5
        assert payload["max_tokens"] == 8000
        assert payload["chat_template_kwargs"]["enable_thinking"] is thinking
        assert ("thinking_token_budget" in payload) is thinking
        return {"choices": [{"message": {"content": "final answer"}, "finish_reason": "stop"}]}

    controller.request = reply
    assert asyncio.run(controller.async_batch_prompt("rules", ["act"])) == ["final answer"]
    assert json.loads((tmp_path / "requests.jsonl").read_text())["sampling"] == sampling


def test_explicit_temperature_override_keeps_other_sampling_fields(tmp_path):
    sampling = sample_configuration()["sampling"]
    controller = Controller("http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, False, 71, sampling=sampling)

    async def reply(payload):
        assert payload["temperature"] == 0.6 and payload["presence_penalty"] == 1.5
        return {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}

    controller.request = reply
    asyncio.run(controller.complete("rules", "act", temperature=0.6))
    assert sampling["temperature"] == 1.0


@pytest.mark.parametrize("invalid", [{"top_k": True}, {"presence_penalty": 3}, {"temperature": float("nan")},
                                   {"top_p": 0}, {"max_tokens": 16000}])
def test_invalid_sampling_is_rejected_before_planning_or_requests(tmp_path, invalid):
    sampling = {**sample_configuration()["sampling"], **invalid}
    with pytest.raises(ValueError, match="sampling"):
        Controller("http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, True, 71, sampling=sampling)
    with pytest.raises(ValueError, match="sampling"):
        run.allocation({**sample_configuration(), "sampling": sampling}, "pilot", 50)


def diagnostic_sources(tmp_path):
    """Three small historical-log fixtures; tests never depend on downloaded experiment data."""
    sources = []
    config = sample_configuration()
    for index, key in enumerate(("action_plan", "possible_opponent_inventory", "action_plan")):
        folder = tmp_path / f"old-{index}" / "episode-0000"
        folder.mkdir(parents=True)
        (folder.parent / "plan.json").write_text(json.dumps({"config": config}))
        game = "pd" if index == 2 else "rws"
        (folder / "summary.json").write_text(json.dumps({"game": game, "language": "en", "thinking": True}))
        row = {
            "model": config["model"], "thinking": True, "sampling_seed": 100 + index,
            "messages": [{"role": "system", "content": "rules" if game == "pd" else
                          "rules\nr_t = transpose(your_inventory) * A_payoff * opponent_inventory"},
                         {"role": "user", "content": f"Return '{key}'"}],
            "response": {"choices": [{"finish_reason": "length", "message": {"content": "unfinished"}}]},
        }
        path = folder / "requests.jsonl"
        path.write_text(json.dumps(row) + "\n")
        sources.append(path)
    return sources


def test_transition_diagnostic_freezes_three_calls_without_allocating_gameplay(tmp_path, monkeypatch):
    import hashlib

    sources = diagnostic_sources(tmp_path)
    output = tmp_path / "diagnostic"
    config = json.loads((sources[0].parents[1] / "plan.json").read_text())["config"]
    monkeypatch.setattr(run, "configuration", lambda: config)
    result = run.plan(output, "diagnostic", 50, replays=sources)
    data = json.loads((output / "plan.json").read_text())
    assert result["replay_requests"] == 3 and data["episodes"] == [] and len(data["tasks"]) == 1
    assert result["check"] == "reasoning_transition"
    assert data["task_wall_minutes"] == [26] and data["maximum_gpu_sbu"] == pytest.approx(83.2)
    assert [case["request"]["sampling_seed"] for case in data["replays"]] == [100, 101, 102]
    assert data["replays"][0]["source_sha256"] == hashlib.sha256(sources[0].read_bytes()).hexdigest()
    assert run.report(output)["status"] == "not_started"


@pytest.mark.parametrize("pd_truncates", [False, True])
def test_transition_diagnostic_runs_three_cases_once_and_preserves_pd_failure(tmp_path, monkeypatch, pd_truncates):
    sources = diagnostic_sources(tmp_path)
    config = sample_configuration()
    output = tmp_path / "diagnostic"
    output.mkdir()
    cases = run.diagnostic_cases(config, sources)
    (output / "plan.json").write_text(json.dumps({"stage": "diagnostic", "config": config, "replays": cases}))
    calls = []

    async def reply(self, payload):
        calls.append(payload)
        truncated = pd_truncates and len(calls) == 3
        key = "action_plan" if "action_plan" in payload["messages"][1]["content"] else "possible_opponent_inventory"
        value = ["move_to((1, 1), (1, 2))"] if key == "action_plan" else {"rock/yellow": 1, "paper/purple": 5, "scissors/blue": 1}
        return {
            "choices": [{"finish_reason": "length" if truncated else "stop",
                         "message": {"content": "unfinished" if truncated else f"```python\n{ {key: value}!r}\n```"}}],
            "usage": {"completion_tokens": 8000 if truncated else 3100,
                      "completion_tokens_details": {"reasoning_tokens": 2999}},
        }

    monkeypatch.setattr(Controller, "request", reply)
    result = asyncio.run(run.replay(output))
    assert len(calls) == 3 and [p["seed"] for p in calls] == [100, 101, 102]
    assert all(p["max_tokens"] == 8000 and p["thinking_token_budget"] == 3000 for p in calls)
    assert result["passed"] == (2 if pd_truncates else 3)
    assert result["ready_for_pilot"] is not pd_truncates
    for index, case in enumerate(cases):
        original = case["request"]["messages"]
        expected = original[0]["content"] if case["game"] == "pd" else language.correct_payoff(original[0]["content"])
        assert calls[index]["messages"][0]["content"] == expected
        assert calls[index]["messages"][1] == original[1]
    assert len(list(output.glob("replays/*/requests.jsonl"))) == 3
    assert run.report(output)["ready_for_pilot"] is not pd_truncates
    if pd_truncates:
        assert "truncated" in result["cases"][-1]["error"]


def test_diagnostic_rejects_mismatched_model_pin_before_creating_allocation(tmp_path, monkeypatch):
    sources = diagnostic_sources(tmp_path)
    config = sample_configuration()
    config["model_revision"] = "different-model-pin"
    monkeypatch.setattr(run, "configuration", lambda: config)
    output = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="pin"):
        run.plan(output, "diagnostic", 50, replays=sources)
    assert not output.exists()


def test_transition_diagnostic_cannot_skip_the_pd_regression(tmp_path, monkeypatch):
    sources = diagnostic_sources(tmp_path)
    folder = sources[-1].parent
    (folder / "summary.json").write_text(json.dumps({"game": "rws", "language": "en", "thinking": True}))
    row = json.loads(sources[-1].read_text())
    row["messages"][0]["content"] = json.loads(sources[0].read_text())["messages"][0]["content"]
    sources[-1].write_text(json.dumps(row) + "\n")
    config = sample_configuration()
    monkeypatch.setattr(run, "configuration", lambda: config)
    output = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="one PD failure"):
        run.plan(output, "diagnostic", 50, replays=sources)
    assert not output.exists()


@pytest.mark.parametrize("missing", [True, False])
def test_transition_diagnostic_requires_native_phrase_before_allocation(tmp_path, monkeypatch, missing):
    sources = diagnostic_sources(tmp_path)
    config = sample_configuration()
    command = list(config["server_command"])
    index = command.index("--reasoning-config")
    if missing:
        del command[index:index + 2]
    else:
        command[index + 1] = json.dumps({"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"})
    config["server_command"] = command
    monkeypatch.setattr(run, "configuration", lambda: config)
    output = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="Transition diagnostic"):
        run.plan(output, "diagnostic", 50, replays=sources)
    assert not output.exists()


@pytest.mark.parametrize("ready", [False, True])
def test_measured_batch_dispatches_transition_diagnostic_and_stops_on_failure(tmp_path, monkeypatch, ready):
    import io

    config = sample_configuration()
    plan = {**run.allocation(config, "diagnostic", 50), "source_hashes": {}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(run, "source_hashes", dict)
    monkeypatch.setattr(run.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        read_text=lambda name: json.dumps({"vcs_info": {"commit_id": config["meltingpot_revision"]}}),
    ))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("COORDINATION_ALLOCATED_SECONDS", "1560")
    terminated = []
    server = SimpleNamespace(poll=lambda: None, terminate=lambda: terminated.append(True), wait=lambda **kwargs: 0)
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: server)
    checks = []

    def models(*args, **kwargs):
        checks.append(True)
        if len(checks) == 1:
            raise OSError("server has not started")
        return io.StringIO(json.dumps({"data": [{"id": config["model"]}]}))

    async def replay(root):
        return {"ready_for_pilot": ready}

    monkeypatch.setattr(run.urllib.request, "urlopen", models)
    monkeypatch.setattr(run, "replay", replay)
    if ready:
        run.batch(tmp_path, 0)
    else:
        with pytest.raises(RuntimeError, match="Transition diagnostic failed"):
            run.batch(tmp_path, 0)
    state = json.loads((tmp_path / "tasks/0000.json").read_text())
    assert (state["status"] == "complete") is ready and state["completed_episodes"] == []
    assert terminated == [True]
    if not ready:
        assert run.report(tmp_path)["task_errors"] and run.report(tmp_path)["status"] == "failed"


@pytest.mark.parametrize("answer", [None, "{'action_plan': ['incomplete']}"])
def test_controller_rejects_token_limit_without_using_reasoning_as_an_answer(tmp_path, answer):
    """Reproduce pilot-07's 4,000 reasoning tokens and length stop, including partial final text."""
    controller = Controller("http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, True, 71)
    response = {
        "choices": [{"message": {"content": answer, "reasoning": "unfinished reasoning"}, "finish_reason": "length"}],
        "usage": {"completion_tokens": 4000, "completion_tokens_details": {"reasoning_tokens": 4000}},
    }

    async def reply(payload):
        return response

    controller.request = reply
    with pytest.raises(RuntimeError, match="truncated.*4000"):
        asyncio.run(controller.complete("rules", "act"))
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    assert row["status"] == "failed"
    assert row["failure_kind"] == "output_token_limit"
    assert row["finish_reason"] == "length"
    assert row["reasoning_tokens_reported"] == 4000
    assert row["response"] == response
    assert controller.calls == 1


@pytest.mark.parametrize(("thinking", "budget"), [(False, None), (True, None), (False, 3000), (True, 3000)])
def test_controller_sends_thinking_budget_only_when_thinking_is_enabled(tmp_path, thinking, budget):
    controller = Controller(
        "http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, thinking, 71, thinking_token_budget=budget
    )

    async def reply(payload):
        assert payload["max_tokens"] == 4000
        assert payload["chat_template_kwargs"]["enable_thinking"] is thinking
        if thinking and budget is not None:
            assert payload["thinking_token_budget"] == budget
        else:
            assert "thinking_token_budget" not in payload
        return {"choices": [{"message": {"content": "final answer"}, "finish_reason": "stop"}]}

    controller.request = reply
    assert asyncio.run(controller.complete("rules", "act")) == "final answer"
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    assert row["thinking_token_budget"] == (budget if thinking else None)
    assert row["max_tokens"] == 4000


@pytest.mark.parametrize("budget", [0, 4000, True, 3.5, "3000"])
def test_thinking_budget_cannot_consume_the_entire_completion_or_have_invalid_type(tmp_path, budget):
    with pytest.raises(ValueError, match="thinking_token_budget"):
        Controller("http://127.0.0.1:8000/v1", "Qwen/test", tmp_path, True, 71, thinking_token_budget=budget)
    config = {**sample_configuration(), "request_tokens": 4000, "thinking_token_budget": budget}
    with pytest.raises(ValueError, match="thinking_token_budget"):
        run.allocation(config, "pilot", episode_minutes=50)


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


def test_report_exposes_saved_task_and_episode_errors(tmp_path):
    data = run.allocation(sample_configuration(), "pilot", episode_minutes=50)
    (tmp_path / "plan.json").write_text(json.dumps(data))
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/0000.json").write_text(json.dumps({"status": "partial", "error": "Episode process exited 1"}))
    spec = data["episodes"][0]
    folder = tmp_path / spec["id"]
    folder.mkdir()
    error = "Completion truncated at max_tokens=4000"
    (folder / "summary.json").write_text(json.dumps({**spec, "status": "partial", "error": error}))
    result = run.report(tmp_path)
    assert result["task_errors"] == {"0000": "Episode process exited 1"}
    assert result["episode_errors"] == {spec["id"]: error}
    assert result["complete"] == 0


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
