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
| Thinking | Qwen off/on with short native action answers | Qwen off/on, separate from HM's ToM algorithm; 8,000-token total with a 3,000-token reasoning cap exercised in pilot-09 |
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

Translation transport also slots fenced Python/JSON examples, numerical literals
(including ordinal spellings such as `2nd`) and machine identifiers. Their exact
source text is restored before freezing; these literals are not language treatment.
Temporary marker indices start after the existing HM expression indices. Both
marker counts and the restored text are structurally validated. JSON/structural
failures allow at most three logged attempts per template within the allocation;
network/server failures still stop the job. This does not certify semantic quality.

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
| Preparation and pilot | 1 H100 × 240 minutes | 768 |
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

Pilot-07 exhausted all 4,000 completion tokens in reasoning and returned no final
content. Pilot-08 exercised vLLM 0.33.0's native `thinking_token_budget: 3000`
within that total: four requests completed, the native RWS agent played 61 frames
and one interaction occurred. Its fifth request reported 2,999 reasoning tokens
and 4,000 total completion tokens. Final content was nonempty but ended mid-word
in the opponent-inventory explanation before the required Python dictionary.
This establishes answer truncation after the reasoning cap took effect; it does
not establish that a larger allowance will finish the answer or correct its math.

Pilot-09 exercised the increased total of 8,000 with the 3,000-token reasoning
cap. This policy applies uniformly across languages and both gameplay modes;
only thinking-on gameplay receives a reasoning cap. Newly generated translations
share the total cap, but all six completed corpora can be reused with provenance.
A null reasoning budget retains the legacy uncapped-within-total policy. Empty
and nonempty length-truncated final outputs remain failures, with full response
and native usage retained; there is no automatic fallback to reasoning or retry.
All earlier failed pilots remain separate from collection data.

### Token-limit investigation

Evidence: frozen `outputs/pilot-0{7,8,9}/source/`, `plan.json`,
`episode-0000/requests.jsonl`, `summary.json` and server logs. Call indices are
zero-based. The existing local runtime replayed responses without model calls;
the actual pinned RWS `extract_dict` method was evaluated independently.

| Pilot | Total / reasoning cap | Successful requests before failure | Native frames | Failed call |
|---|---|---|---|---|
| 07 | 4,000 / uncapped | 1 | 0 | 1: all 4,000 tokens in reasoning, no final content |
| 08 | 4,000 / 3,000 | 4 | 61 | 4: nonempty explanation cut off before inventory dictionary |
| 09 | 8,000 / 3,000 | 14 | 93 | 14: repeated planning text, no action dictionary |

These are not controlled causal comparisons: pilot-08's initial player position
and orientation differ from pilots 07/09 despite identical recorded seeds. The
initial prompts in 07 and 09 are identical. The origin of the 08 environment
variation is unverified; replay exact saved messages rather than assuming a new
episode recreates a particular failed prompt.

**Confirmed mechanism.** Pilot-09's failed response has 16,704 final-content
characters. The same planning paragraph begins 33 times; a full identical
339-character block occurs 31 times. The reasoning field ends midway through
the reward equation and the content field continues that sentence. Fourteen of
15 responses report 2,999 reasoning tokens. The cap closes the reasoning region
but does not ensure the model starts a useful final answer. vLLM documents forced
reasoning termination and an optional transition phrase in
[`reasoning_end_str`](https://docs.vllm.ai/en/v0.33.0/features/reasoning_outputs/#thinking-budget-control).
That phrase is a possible later intervention, not an already-tested fix.

**Most actionable configuration mismatch.** Pilot-09's frozen `controller.py`
inherits temperature 0.2 and top-p 1.0 from HM's GPT-4 configuration in upstream `main.py`. The selected
agents/runners supply no temperature overrides. The actual server starts with
`--generation-config vllm`, which
[disables the checkpoint generation configuration](https://docs.vllm.ai/en/v0.33.0/cli/serve/#--generation-config).
Omitted presence penalty and top-k therefore resolve to 0 and disabled filtering,
respectively, under the pinned
[chat protocol defaults](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py).
The [pinned Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md#best-practices)
recommends different settings and identifies presence penalty as a repetition
control. This makes sampling a source-backed hypothesis to test, not proven
sole causation from these observational logs.

**Upstream prompt defect.** The RWS system message gives the payoff as the raw
inventory bilinear product, while its numerical examples require division by
both inventory sums. The pinned Melting Pot `the_matrix/components.lua` resolves
interactions by normalizing both inventories before multiplying. For the prompt's
two examples, raw products are -810 and 160; normalized values are -5.625 and
3.265306. Saved reasoning explicitly struggles with this discrepancy. The prompt
also requests potentially long step-by-step prose before the Python dictionary.
These are plausible contributors, not proof that either alone causes repetition.
Do not silently rewrite pinned prompts: localization hashes and translations
would need a documented revision.

**Failure propagation, not 18 independent failures.** The upstream parser returns
an empty dictionary for the saved failed answers, because neither contains its
required code block. Accepting truncated output would not recover a plan. The
controller raises; the episode process fails; `batch` stops at its checked
subprocess call. Seventeen episodes remain unattempted. Pilot-09's last prompt
uses 6,219 tokens, so input plus the 8,000-token allowance is 14,219, below the
16,384-token server context. All 15 API calls return HTTP 200; server termination
follows batch cleanup. These logs show output exhaustion, not context overflow,
a Slurm time limit or a pre-response GPU out-of-memory failure. They do not prove
the absence of every possible model/runtime numerical issue.

**Authorized sampling revision, evaluated in diagnostic-01.** After the user
authorized the reasonable fixes, retain 8,000 total and 3,000 reasoning tokens.
Use one explicit common sampling profile for both thinking
conditions: temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 1.5,
repetition penalty 1.0. This is Qwen's thinking/general-task recommendation; using
the same profile in both conditions avoids confounding the thinking contrast with
different sampler settings, but is a deliberate study choice. Qwen warns that
penalties can also affect performance or language mixing. Freeze and log all six
fields, and preserve translation provenance. The local controller now forwards
the configured profile through both direct and HM batch calls, preserving an
explicit caller temperature override. Planning and construction reject invalid
sampling before requests. Old plans without the field retain the six explicit
legacy defaults. Changing server defaults alone would be insufficient because
the old client explicitly overrides temperature and top-p.

**Diagnostic-01 outcome.** Job 29284608 completed all six sampler-comparison
requests in 7:19. Legacy sampling passed two cases; Qwen sampling passed one.
Both solved the initial action request. The inventory request passed only under
legacy sampling; the later action request failed under both. All three failures
reported 2,999 reasoning / 8,000 completion tokens and the pinned parser recovered
no usable dictionary. The Qwen inventory response repeatedly tries to reconcile
fractional rewards with the unnormalized integer equation. The Qwen later-action
response contains three backtick sequences inside a formatting discussion, not
a valid required action dictionary. The sampler change is not established as a
fix, nor do three prompts establish the generally better sampler.

**Normalized-payoff correction.** The next revision corrects the demonstrated
equation defect, without also changing the reasoning termination mechanism.
`language.correct_payoff` replaces exactly one literal equation after localization
for RWS/repeated and RWS/arena, in both thinking modes. The replacement is
`r_t = (transpose(your_inventory) * A_payoff * opponent_inventory) / (sum(your_inventory) * sum(opponent_inventory))`.
All twelve frozen system prompts (two games, six languages) contain the same
original expression; offline checks verify that only that line changes. The
original corpus bytes, prose and language instructions remain intact. Episode
manifests record `prompt_corrections: ["rws_payoff_normalization"]`, while raw
requests retain the actual corrected messages. This is an explicit departure
from the upstream prompt, matching the already-pinned native payoff; it does not
change native rewards, actions or other game rules. Missing or ambiguous equations
are rejected before inference. There is no correction in Cooking or PD.

Diagnostic-02 tested original versus normalized system prompts
at identical current sampling, user messages, seeds and token caps. Its plan
declares `diagnostic_comparison: rws_payoff_normalization` and freezes each full
historical failed request, the original log SHA-256 and the normalized system
message, verifying model and implementation pins before allocation creation.
This replaces the earlier sampler comparison for new allocations; diagnostic-01
and its frozen source/outputs remain unchanged and readable. Those reports use
`original_passed` / `normalized_passed`, rather than legacy/Qwen labels.

The resident server and measurement wrapper ran all six requests once.
The wall ceiling was startup plus 20 minutes (six 180-second request timeouts
plus two cleanup minutes), currently 35 minutes / 112 GPU-SBU. A transport error,
truncated response or missing required upstream dictionary key is saved as failed.
Only three successful normalized cases yielded `ready_for_pilot`; a failed corrected
case made the batch fail. Startup errors remain visible in `report`. Diagnostic
results are not study data and do not certify game-action validity or language
quality. CPU tests/replays cannot establish that this correction stops generation
loops. The separate reasoning-transition intervention remains unevaluated.

**Diagnostic-02 terminal result.** The supplied report for job 29285436 shows
completion with exit 0 in 6:32: all three normalized prompts passed, while the
original prompts passed one of three. The two original failures still exhausted
8,000 completion tokens. This supports the correction for these exact saved
English thinking-on requests and meets the diagnostic gate for the full pilot.
It does not establish general reliability, valid executed actions, non-English
performance or collection timing. The downloaded evidence was subsequently
reviewed locally: each pair has identical user messages, sampling seeds, sampler
fields and token caps, with exactly the equation line changed in the system
message. All corrected responses ended with `stop` at 3,346, 3,838 and 3,271
completion tokens, and their saved parsed results match the pinned parser.
NVML measurement completed with exit 0 and positive whole-window GPU energy
(0.0321071 kWh); CodeCarbon 3.3.2 recorded positive GPU energy separately, and
EcoLogits 0.11.3 accounted for six requests. These overlapping energy scopes
must not be added together.

Only after a passing diagnostic should the full pilot establish cross-game,
multilingual and timing feasibility. Pilot-09 spent about 476.5 seconds between
its first request and final failure for only 93 frames; the 50-minute collection
episode ceiling remains a limit, not a validated runtime forecast.

**Pilot-10 (job 29285711).** English RWS completed 200 frames with thinking on
(40 requests, four interactions, 1,030.875 seconds episode wall time) and off
(33 requests, three interactions, 632.654 seconds). Summed request durations are
1,215.61 and 630.43 seconds respectively and must not be used as episode elapsed
time because HM can issue concurrent requests. English PD/thinking-on reached
63 frames and one interaction before call 19 exhausted 8,000 completion tokens,
including 2,999 reported reasoning tokens. Its input was 6,304 tokens, below the
16,384-token context with the requested output allowance. The model claims target
(14, 5) is missing from valid locations, but offline parsing confirms it is present.
The final response continues unfinished reasoning, has no dictionary fence and
cannot be parsed. This is not evidence for changing perception or the map.
Both completed RWS records remain intact; the batch stopped before the other
15 episodes. The whole-job measurement is appropriately partial/failed.

**Reasoning-transition revision.** The next smallest intervention uses the exact
vLLM 0.33.0 documented `reasoning_end_str` transition phrase before `</think>`;
the server command freezes it with explicit `--reasoning-config`. The model,
runtime, 8,000 total / 3,000 reasoning limits, sampler and RWS equation correction
are retained. This fixed English phrase is an internal forced completion of the
reasoning region, applied across languages in the thinking-on condition only;
its generated tokens and any language effects belong to that treatment. It is
not an extra unlogged model call and does not modify the native game state.
Live effectiveness and multilingual behavior remain unverified.

The current diagnostic is a three-request acceptance check: RWS inventory and
action failures from pilots 08/09, plus the PD action failure from pilot-10. It
requires exactly two RWS cases and one PD case so the new regression cannot be
omitted. Each source retains its original log SHA-256, selected full failed row,
game, exact user message and sampling seed; existing RWS corrections are applied
before replay, while PD's system prompt is unchanged. Each case uses its game's
pinned parser. The plan declares `diagnostic_check: reasoning_transition` and
freezes the server transition configuration. There are no fresh paired controls
or retries. Prior diagnostic-01/02 reports and source snapshots remain readable.
Startup plus eleven minutes (three 180-second HTTP timeouts and two cleanup
minutes) gives 26 minutes / 83.2 GPU-SBU. Only `passed: 3` yields readiness for a
full pilot. Missing native transition configuration is rejected before allocation
creation. Diagnostic success does not qualify collection or prove general
generation reliability. Full-horizon runtime and the 256-request ceiling need
review before spending the collection allocation; no horizon or factor reduction
has been made to accommodate the observed rates.

**Diagnostic-03 terminal result and next pilot.** The supplied report for job
29287560 shows all three replay cases passed, including PD's previously failed
move to (14, 5), with no task error. Downloaded evidence was reviewed locally:
all three responses ended with `stop`, matched their pinned game parsers, and
retained the intended messages and sampling seeds. The transition phrase appears
in the reasoning field, not final content. Native reasoning usage is 3,012 tokens
in each case: the forced transition adds tokens after the configured 3,000-token
threshold. Total completion counts are 3,841, 3,308 and 3,254, all below 8,000;
do not clamp or subtract forced tokens from reported usage. Measurement completed
with exit 0 and positive NVML whole-window GPU energy (0.01335085 kWh).
CodeCarbon 3.3.2 separately records positive GPU energy, and EcoLogits 0.11.3
accounts for three requests. This meets the request-level gate for the next full pilot, not the
collection gate. The pilot allocation ceiling is now four hours / 768 GPU-SBU
because pilot-10's two completed episodes consumed about 28 minutes before PD.
The 18 pilot conditions, seeds, 200/100-frame horizons and 50-minute per-episode
timeout are unchanged. This is an operational ceiling increase, not an observed
four-hour runtime or a new model intervention. With the current 15,744-SBU
collection ceiling and 2,031.9067-SBU private reserve, the pre-pilot effective
balance must exceed 18,543.9067 SBU, excluding any other outstanding jobs. The
next pilot is still a feasibility check: no claim that 1,500-frame collection
episodes fit 50 minutes follows from three replay responses.

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
