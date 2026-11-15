# Migration and study contract

15 November 2026. This study uses HM-supported games and HM's original focal-agent
protocol. Account balances and reserved funds are read from the ignored local
`.budget.yaml` file. They are not published. No compute jobs are submitted by
repository extraction or local verification.

## What changes, and why

| Requirement | Previous native controller | This HM study |
|---|---|---|
| Agent decisions | LLM chooses a native action each active frame | Published HM hypotheses, high-level plans, subgoals and deterministic action functions |
| Games | Allelopathic Harvest, Clean Up, Coins, forced Cooking | Asymmetric Cooking, repeated PD, repeated RWS, RWS arena |
| Population | Every player is an LLM | One focal LLM, original scenario background bots |
| Language | Six fully rendered interfaces | Static HM prose translated before gameplay; generated actor text remains in its condition language |
| Memory | Last ten own-action/reward frames | HM's original spatial/episodic and hypothesis state, reset each episode |
| Thinking | Qwen off/on with short native action answers | Qwen off/on, separate from HM's ToM algorithm; HM's 4,000-token response cap |
| Environment | Melting Pot 2.4.1 and custom observation adapters | Pinned HM Melting Pot 2.2.0 fork and original sprite-based perception |
| Measurements | EcoLogits/NVML; older CodeCarbon pilot | CodeCarbon totals plus reused NVML phases and separate EcoLogits records |

The new dataset cannot be pooled with the old dataset as if only language changed.
Games, population composition, memory, action abstraction and inference workload
all differ. Old data and the KG project remain in their original directory;
`history/extraction.json` records which original files were consulted.

## Reuse versus migration complexity

Retaining the exact old four games would require new HM agents, prompt families,
state-to-action mappings and validation. Accepting the upstream game set avoids
that work. Retaining all-LLM self-play would require changing the published
episode loops and opponent bookkeeping. Accepting the original one-focal-agent
protocol avoids that work too.

What remains local is the part specific to this contribution: language treatment,
Qwen connection, experimental allocation, provenance and environmental accounting.
The original generic Ray/Gym tuple-conversion layer is replaced with `hm.Scenario`,
which uses the fork's official scenario factory. This avoids Ray, Gym and RLlib
dependencies. At load time `language.load_module` changes exactly the upstream
`import llm_plan.env.utils as utils` import to `import hm as utils`. The scenario
factory, background policies, action order and HM perception/decision methods are
retained. All four English agent initializations are compared with their original
classes in the local verification.

One upstream execution bug is corrected at load time: the PD runner's
`while step < 1200` loop becomes `while not done`. It otherwise ignored native
termination and the configured frame cap. A regression check reproduces the old
1,200-step behaviour and verifies the corrected stop condition. All four original
HM episode loops were also executed for two native frames with fixed offline
responses. These checks validate integration, not Qwen task competence.

Upstream code remains a pinned, unmodified submodule. Updating it requires a new
source inventory and pilot. Do not collapse the five small local modules into one
file: allocation, localization, HTTP calls, environment and instrumentation have
different verification boundaries.

## Language treatment

`language.Text` inventories prose string literals and f-strings from the selected
four agents and their episode loops. The initial source has 74 unique templates,
about 12,068 English words. Translation deduplicates repeated wording.
Expression markers such as `⟦0⟧` stand for existing Python expressions; translated
text can reorder them but cannot introduce code. Original dictionary keys,
comparisons, parser delimiters and logging-only strings are not translated.
Native object IDs, function names, numerical values, coordinates and Python/JSON
output keys remain stable machine-interface symbols. This distinction is part of
the treatment and must be described in the paper.

Translation is a separate measured preparation phase. Generated templates are
validated and frozen; the planner copies them into collection. No live translation
or additional translation service is inserted into the gameplay loop. Generated
HM strategy/hypothesis text is retained as the model produced it; language drift
is visible in the raw responses, rather than hidden by post-hoc rewriting.

Structural checks preserve placeholders, numerical tokens and quoted protocol
identifiers. They do not establish meaning equivalence or rule fidelity in all
languages. Fluent review remains an acknowledged limitation until reviewers set
the corresponding corpus `reviewed` fields after inspection. Collection remains
explicitly exploratory; no confirmatory or culture-level claim is implied.

## Minimal verification and budget

Local checks require no model or paid GPU: allocation arithmetic/balance, invalid
budgets, source/template round trips, unchanged parser keys, raw request accounting,
partial-episode reporting, native reward accounting and energy units. The bounded
live job then checks the actual H100, server, fork, complete language inventory,
native gameplay and all three measurement tools together.

| Allocation | Calculation | Maximum GPU SBU |
|---|---|---:|
| Preparation and pilot | 1 H100 × 120 minutes | 384 |
| Collection example | 8 blocks × (15 + 12 × 50) minutes | 15,744 |
| Reserved contingency | Set privately in `.budget.yaml` | Local value |
| Remaining allowance | Local balance minus allocation ceilings and reserve | Computed locally |

The rate is 192 SBU per H100-hour, inherited from the reviewed SURF accounting
reference. CPU staging charges, outstanding accounting updates and any unrelated
jobs must also fit the balance. A wall cap bounds spending; it does not guarantee
episode completion, useful behaviour, statistical power or enough context length.
The 96 episodes supply two exploratory replicates per condition, not a powered
confirmatory sample.

The pilot covers all four games and both modes in English, plus both modes in
the other languages on Cooking. It does not exhaustively test every game/language
combination. This is an explicit compromise to avoid a separate expensive testing
campaign. It may reveal that the planned run is not affordable; preserve that
outcome and revise the design before collection rather than spending the reserve
on automatic retries.

## Outcomes, units and inference

### Cooperation versus reward

The related payoff-scaling paper defines cooperation as the share of an agent's
ten binary decisions that select the cooperative option. HM's spatial PD instead
uses a mixed strategy defined by the normalized resource inventory at a completed
interaction. In the pinned native configuration resource 1 is green/cooperate and
resource 2 is red/defect; Lua normalizes each inventory before applying the payoff
matrix. We therefore record `green / (green + red)` at each native interaction,
then take the unweighted within-episode mean as `pd_cooperation_share`.

`pd-cooperation.jsonl` retains the raw inventories, focal/partner shares and frame.
`pd_interactions` records the denominator; zero-interaction episodes have a missing
cooperation share, not zero cooperation. The same rule applies regardless of which
player is the row player. This is an inventory-based cooperation measure, not an
identical replication of binary Option A/B decisions. Language can affect whether
an interaction is reached, so this conditional measure is reported alongside the
interaction count and native returns, rather than replacing total outcomes.

Cooking supplies coordination/task-completion measures. RWS and RWS Arena are
competitive games and receive no generic cooperation score. Higher return alone
is never labelled more cooperation. Native payoffs stay fixed; this study does
not add the related paper's payoff-scaling or persona treatments.

Sources: [payoff-scaling paper, methods](https://arxiv.org/html/2601.19082v3),
[HM PD methods](https://arxiv.org/html/2407.07086v2),
[native PD configuration](https://github.com/locross93/meltingpot/blob/0438de9506db218af39e144c671fc697c14ac6c3/meltingpot/configs/substrates/prisoners_dilemma_in_the_matrix__repeated.py)
and [native interaction calculation](https://github.com/locross93/meltingpot/blob/0438de9506db218af39e144c671fc697c14ac6c3/meltingpot/lua/levels/the_matrix/components.lua).

### Experimental and measurement units

The experimental unit is the independent population episode containing one focal
agent. The fixed scenario contributes a background-policy distribution; it is not
a guarantee of identical partners or trajectories across conditions. Environment
and sampling seeds have separate namespaces, and every episode starts fresh.
Blocks contain all twelve language × thinking cells for one game and replicate.
Their order and the game order are randomized from the frozen allocation seed.

Record focal native return, full native-population summed/per-capita returns,
per-player returns, native events/actions and game-specific task events. Background
behaviour is not an independent replicate, and whole-population returns include
the background policies. These are effects of changing the focal LLM's interface,
not effects of making an entire population multilingual. Report within game;
do not pool native reward scales or interpret raw signed-return Gini.

HM's original perception reads WORLD.RGB and forms its ego/window/memory
representation, with its original map knowledge. This differs from the old strict
local-RGB adapter. The new evaluator subscriptions additionally retain all native
actions, rewards and events, but never feed them into the focal agent's prompt.
The fork's original observation permissions are preserved; no transformed/training
scenario is substituted for evaluation.

Per-request logs distinguish input, total completion and server-reported reasoning
tokens, finish reason, failures, condition, seed and wall time. The upstream HM
display counter is retained as zero for interface compatibility with local models;
the exported monetary cost is null. It is not evidence of free computation.
Concurrent request durations are summed only in the explicitly named request-time
field, not treated as elapsed episode time.

Measured episode GPU energy sums environment-setup and gameplay phase counter
deltas. Server startup, translation and between-episode overhead remain separately
visible. NVML power maxima are sampled values, not continuous peak measurements.
CodeCarbon supplies whole-job totals and its own CPU/RAM attribution estimates;
these are not allocated to individual hypotheses or agents as direct measurements.
EcoLogits supplies separate supported-model estimates or explicit unavailable
statuses. Unknown usage/counters stay missing. Never add overlapping energy scopes.

No final covariance estimator, contrast family or power calculation has been
chosen for this changed protocol. Report exploratory episode-level effects with
the design limitations; freeze any later confirmatory analysis separately.
