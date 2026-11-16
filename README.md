# LLM language coordination

Evaluate **Hypothetical Minds (HM)** in six languages while recording native game
behaviour, inference requests, GPU energy and estimated emissions. This repository
is independent of the graph-memory project. It does not contain the KG code or
the earlier custom action-by-action controller.

The accepted protocol uses **one focal Qwen3.5-9B agent against HM's background
partners**, four HM-supported games, English/Dutch/French/Spanish/Arabic/Chinese,
thinking off/on and two independent episodes per condition: **96 exploratory
collection episodes**, each capped at 1,500 native frames. Scenario 0 is selected
in each game; this is not a replication of every scenario in the HM paper.

HM's original agent classes, hypotheses, memory, pathfinding and episode loops
come from a pinned Git submodule. Our code supplies localization, a local Qwen
connection, official-scenario conversion and measurement. See
[the migration and study contract](docs/migration.md) and [sources](SOURCES.md).

RWS and arena system prompts receive one documented correction after localization:
the reward equation divides by both inventory sums, matching the native game and
the existing numerical examples. It applies to every language and both thinking
modes. Original corpus files remain intact; episode manifests identify the correction.

## Structure

| File | Responsibility |
|---|---|
| `experiment.yaml` | Factors, checkpoint pins, horizons, allocation ceilings and server command |
| `run.py` | Freeze allocations, run episodes, summarize saved evidence |
| `language.py` | Extract HM prose and freeze translations without changing Python expressions |
| `controller.py` | HM's request interface, raw responses, native token usage and errors |
| `hm.py` | Official scenario adapter and evaluator-only full-population returns/events |
| `measure.py` | Reused NVML sampler, CodeCarbon totals and separate EcoLogits estimates |
| `job.sbatch` | One H100 job using the exact frozen wall time |
| `vendor/hypothetical-minds/` | Unmodified upstream dependency and sprite assets |

## Get the source

```bash
git clone --recurse-submodules https://github.com/glp500/llm-language-coordination.git
cd llm-language-coordination
# For an existing checkout:
git submodule update --init --recursive
```

Use Miniforge/Conda and the existing Snellius Apptainer image. Do not rebuild the
working Qwen/vLLM environment. HM requires its older Melting Pot fork, so it gets
a separate Conda environment. Native episodes run inside the image because the
Snellius host glibc cannot load Lab2D's wheels.

Create an ignored `.budget.yaml` with your current numeric `balance_sbu` and
`reserve_sbu` values before planning. Account figures and machine-specific paths
belong in local files; do not commit them. The public configuration leaves these
values unset. The planner checks each allocation plus the pilot ceiling and your
reserve against that local balance.

Run the following installation on a CPU staging allocation, after connecting
the existing image, inference environment and model cache at the three paths
below. These are links to existing resources, not copies of the old project.
Use your actual paths if the earlier run directories differ.

```bash
ln -s /path/to/existing/harvest.sif harvest.sif
ln -s /path/to/existing/inference-environment .venv
mkdir -p .cache logs
ln -s /path/to/existing/model-cache .cache/huggingface
mamba env create --prefix "$PWD/.conda/runtime" --file environment.yml
conda activate "$PWD/.conda/runtime"
apptainer exec harvest.sif "$CONDA_PREFIX/bin/python" -m pip install -r requirements.txt
apptainer exec harvest.sif "$CONDA_PREFIX/bin/python" -m pip check
"$CONDA_PREFIX/bin/python" -m pytest -q
```

Use the Conda prefix's Python explicitly after loading modules: reactivating an
inherited environment can leave Miniforge's base Python first on `PATH` while
`CONDA_PREFIX` still names the project environment. `job.sbatch` uses the explicit
interpreter for accounting, measurement and the batch runner. If a job fails,
preserve its allocation and logs; create a new pilot directory for another attempt.

The image and `.venv` link are **runtime-only resources**, ignored by Git. The new
repository does not import code from the KG project. The pinned Melting Pot
installer downloads approximately 463 MB of upstream background-policy assets;
perform that preparation before requesting a GPU. The model weights must already
be in the reused cache at the pinned revision. Installation commands do not submit
jobs. Tests use no model calls or Snellius GPU time.

## One bounded preparation/pilot job

The current `request_tokens: 8000` and `thinking_token_budget: 3000` were exercised
in pilot-09. That run completed 14 requests and 93 native frames, then repeated
the same planning paragraph until its answer allowance was exhausted. Increasing
the total cap alone did not resolve generation stability. See the
[pilot-07 through pilot-09 investigation](docs/migration.md#token-limit-investigation)
before spending another GPU allocation. Following that review, `sampling` now
explicitly sets temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty
1.5 and repetition penalty 1.0. The same profile is used in both thinking modes
and logged per request. Frozen plans without `sampling` retain legacy defaults.

The total cap applies to both gameplay modes and newly generated translations.
Only thinking-on gameplay receives the reasoning cap; frozen translations are
reused. A `null` reasoning cap preserves uncapped-within-total behaviour. This
is a capped-thinking condition, frozen in the plan and request logs. Truncated
answers remain failures; do not remove that guard to make a run appear complete.

The current `diagnostic` stage freezes three English thinking-on requests: two
RWS cases and the PD failure from pilot-10. Diagnostic-01 compared samplers
(legacy 2/3, Qwen 1/3); diagnostic-02 established the RWS equation correction on
its three cases. Pilot-10 completed both 200-frame RWS episodes, then failed on PD.

This check uses vLLM's documented transition phrase before `</think>` when the
3,000-token reasoning budget runs out. It retains 8,000 total tokens, the sampler,
native prompts and the established RWS equation correction. The fixed English
transition is appended when the configured reasoning threshold is reached and
is part of the declared thinking-on treatment across languages. Its tokens also
appear in native reasoning usage: diagnostic-03 reported 3,012 reasoning tokens
at the 3,000-token threshold. Preserve that usage; the 8,000-token total cap still
applies. Thinking-off requests have no reasoning budget and
do not trigger the forced phrase. Each case is attempted once under the new
configuration, using the same server, measurement wrapper and task guards. This
is an acceptance check against historical failures, not a fresh paired control.
Startup plus eleven request/cleanup minutes gives a **26-minute, 83.2 GPU-SBU
ceiling**. No translations or environment episodes run.

```bash
"$PWD/.conda/runtime/bin/python" run.py plan outputs/diagnostic-03 --stage diagnostic --replay \
  outputs/pilot-08/episode-0000/requests.jsonl \
  outputs/pilot-09/episode-0000/requests.jsonl \
  outputs/pilot-10/episode-0002/requests.jsonl
sbatch --job-name=language-diagnostic-03 --time=00:26:00 job.sbatch "$PWD/outputs/diagnostic-03" 0
# After the job ends:
"$PWD/.conda/runtime/bin/python" run.py report outputs/diagnostic-03
```

The report must show `passed: 3` and `ready_for_pilot: true` before considering
another full pilot. All three cases are attempted once; a failure makes the batch
fail. Historical diagnostic results remain intact and readable.
`ready_for_pilot` checks completed nontruncated output and the required upstream
dictionary key, not game competence, multilingual quality or collection readiness.
Diagnostic results are not study episodes. There is no automatic resubmission.

```bash
"$PWD/.conda/runtime/bin/python" run.py plan outputs/pilot-11 --stage pilot
sbatch --job-name=language-pilot-11 --time=04:00:00 job.sbatch "$PWD/outputs/pilot-11" 0
```

The job starts Qwen once, translates **74 unique HM prose templates** into the
five other languages, checks their expression markers/numbers/protocol IDs, then
runs 18 short episodes: all four games in English with thinking off/on, and
Cooking in each other language with thinking off/on. The entire job, including
translation, loading and retries, has a **768 GPU-SBU ceiling**. The four-hour
allowance follows pilot-10's measured RWS timings; it is not a completion forecast.
The episode horizons and research conditions are unchanged. It has no automatic
resubmission. Preserve partial output if it fails; a failure is not zero energy.

Preparation allows at most three attempts per template when JSON or structural
validation fails. Each retry names the validation error; raw requests remain in
`preparation/requests.jsonl`, with rejection reasons and request indices in
`preparation/translation-validation.jsonl`. Invalid wording is never frozen.
Network/server failures stop the job. Retry time stays within the same allocation.
Before translation, code examples, numerical literals (including `2nd`-style
ordinals and complete grid dimensions such as `23x15`) and protocol identifiers
are replaced by temporary expression-style markers. Their exact text is restored
before the original structural validation and freezing; existing HM expression
markers remain separate and unchanged.
Numeric validation counts literals regardless of adjacent script, including both
dimensions in `23x15` and numerals joined to Arabic or Chinese prose.
Range separators such as `3-5` may be translated as "3 to 5"; genuine negative
payoff literals retain their signs and changed endpoints are rejected.

A fresh pilot may reuse structurally valid corpus files copied into its
`translations/` before submission; preparation validates and skips saved entries.
Keep the donor allocation and hashes of its corpus/request files. Translation
generation energy remains in that donor attempt's measurements.

Translations are generated once and saved under the pilot's `translations/`.
There are no translation calls during collection gameplay. Their `reviewed`
field starts false for non-English text: structural checks do not certify fluent
semantic equivalence. Inspect the frozen wording and report provisional status.

## Freeze collection after inspecting the pilot

```bash
"$CONDA_PREFIX/bin/python" run.py report outputs/pilot-01
"$CONDA_PREFIX/bin/python" run.py plan outputs/collection-01 --stage collection \
    --pilot outputs/pilot-01 --episode-minutes 50
```

Collection planning refuses missing/failed pilot episodes, changed execution
sources, missing translations, missing GPU energy, missing CodeCarbon data or
missing EcoLogits accounting. A completed short pilot still does not prove task
competence or predict full-horizon runtime. Inspect `episodes.csv`, the native
events, requests and frame images before spending the collection allocation.

The 50-minute example creates eight blocks of 12 episodes. Each block has a
15-minute server-startup allowance: **10:15:00 per job**, **15,744 GPU SBU total**.
Do not submit this allocation if the pilot suggests that 1,500-frame episodes
will exceed the chosen cap. Increasing the cap is checked against the budget;
reducing replication/games is a new design decision, not an automatic fallback.

To run the frozen example sequentially, submit the first block, then submit each
next block with `--dependency=afterok:PREVIOUS_JOB_ID`. This bounds concurrency
to one GPU and stops subsequent spending after a failed block:

```bash
sbatch --job-name=language-collection --time=10:15:00 \
    job.sbatch "$PWD/outputs/collection-01" 0
# After a successful first block, replace PREVIOUS_JOB_ID and increment the index:
sbatch --job-name=language-collection --time=10:15:00 \
    --dependency=afterok:PREVIOUS_JOB_ID job.sbatch "$PWD/outputs/collection-01" 1
# Continue through index 7, retaining each dependency.
"$CONDA_PREFIX/bin/python" run.py report outputs/collection-01
```

Use the exact wall times printed by the planner if you change the episode cap.
`batch` refuses a Slurm allocation whose wall time differs from its frozen plan.
Existing task/output directories cannot be reused for another attempt.

## Evidence and measurements

Every episode retains the condition/seeds, full prompts and responses, returned
token usage, request timing/failures, HM plans/hypotheses, native population
actions/rewards/events and original frame images. Evaluator-only background
rewards are recorded separately from the actor's input. Missing/partial episodes
remain in the report. Reported reasoning tokens stay missing if the server does
not supply them; total completion tokens still include generated reasoning.
Length-truncated completions fail explicitly even if some final text is present;
reasoning is never substituted for a final plan. Reports include saved task and
episode error messages, and request logs retain finish reason, token limits and
sampling settings for failures as well as successes.

PD additionally records **cooperation share at each completed interaction**:
green resources divided by green plus red resources. Reports include its episode
mean, the partner's share and the number of interactions. No interaction means
missing cooperation, rather than defection. This adapts cooperation measurement
to HM's mixed-strategy game; it is not the payoff-scaling paper's binary-choice
protocol. Cooking coordination and competitive RWS outcomes remain separate.

`energy/` keeps CodeCarbon CSVs, approximately one-second NVML power/memory traces,
phase counters, host/process memory/utilization, environment versions, job IDs
and EcoLogits status records. CodeCarbon's job totals, measured GPU phases and
EcoLogits request estimates describe different scopes: **never add them**.
Qwen3.5 is absent from the pinned EcoLogits model registry, so its request-impact
estimate is explicitly unavailable. GPU energy comes from NVML; operational GPU
emissions use the recorded EcoLogits electricity coefficient. CPU/RAM energy in
CodeCarbon may be estimated. Cooling and embodied hardware impacts are excluded.

No GPU model, translated corpus or scientific result is bundled with this source.
The runtime and translation pilot must pass on your allocated H100 before collection.
