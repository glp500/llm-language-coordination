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
python -m pytest -q
```

The image and `.venv` link are **runtime-only resources**, ignored by Git. The new
repository does not import code from the KG project. The pinned Melting Pot
installer downloads approximately 463 MB of upstream background-policy assets;
perform that preparation before requesting a GPU. The model weights must already
be in the reused cache at the pinned revision. Installation commands do not submit
jobs. Tests use no model calls or Snellius GPU time.

## One bounded preparation/pilot job

```bash
python run.py plan outputs/pilot-01 --stage pilot
sbatch --job-name=language-pilot --time=02:00:00 job.sbatch "$PWD/outputs/pilot-01" 0
```

The job starts Qwen once, translates **74 unique HM prose templates** into the
five other languages, checks their expression markers/numbers/protocol IDs, then
runs 18 short episodes: all four games in English with thinking off/on, and
Cooking in each other language with thinking off/on. The entire job, including
translation, loading and retries, has a **384 GPU-SBU ceiling**. It has no automatic
resubmission. Preserve partial output if it fails; a failure is not zero energy.

Translations are generated once and saved under the pilot's `translations/`.
There are no translation calls during collection gameplay. Their `reviewed`
field starts false for non-English text: structural checks do not certify fluent
semantic equivalence. Inspect the frozen wording and report provisional status.

## Freeze collection after inspecting the pilot

```bash
python run.py report outputs/pilot-01
python run.py plan outputs/collection-01 --stage collection \
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
python run.py report outputs/collection-01
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
