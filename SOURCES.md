# Sources and reuse

Source review: 15 November 2026. Links describe related work or implementation
provenance; an inspiration paper is not automatically an implemented algorithm.

## Directly applied

- Cross, L., Xiang, V., Bhatia, A., Yamins, D. L. K., and Haber, N. (2024).
  [Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models](https://arxiv.org/abs/2407.07086).
  [Pinned code](https://github.com/locross93/Hypothetical-Minds/tree/e1788c051608099f4b89c63d19de889a3706e36b).
  Actual HM agent classes, ToM, memory, action functions, sprite perception and
  episode loops. The dependency is MIT licensed; its copyright and license remain
  in `vendor/hypothetical-minds/LICENSE`.
- Agapiou, J. P., et al. (2022; revised 2023).
  [Melting Pot 2.0](https://arxiv.org/abs/2211.13746).
  HM's [pinned fork](https://github.com/locross93/meltingpot/tree/0438de9506db218af39e144c671fc697c14ac6c3)
  supplies the actual scenarios, background policies and original observation
  permissions. It is Apache-2.0 licensed.
- Qwen Team. [Pinned Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
  and [model card](https://huggingface.co/Qwen/Qwen3.5-9B).
  Actual focal model, tokenizer/chat template and explicit thinking control.
- [vLLM serving documentation](https://docs.vllm.ai/en/latest/cli/serve/).
  The existing project's frozen vLLM 0.33.0 runtime is reused, including its
  documented `--disable-prefix-caching` spelling and Qwen reasoning parser.
- NIST/SEMATECH: [randomized blocks](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)
  and [factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm).
  Basis for language/mode combinations and allocation blocking.
- [CodeCarbon 3.3.2 source](https://github.com/mlco2/codecarbon/tree/v3.3.2),
  [API/configuration](https://docs.codecarbon.io/latest/reference/api/),
  [methodology](https://docs.codecarbon.io/latest/explanation/methodology/) and
  [output definitions](https://docs.codecarbon.io/latest/reference/output/).
  Whole-job measurement/estimation. The old bibliography's
  [Zenodo record](https://zenodo.org/records/11171501) identifies version 2.4.1,
  so it must not be presented as the executed 3.3.2 release.
- Rincé, S., and Banse, A. (2025).
  [EcoLogits: Evaluating the Environmental Impacts of Generative AI](https://joss.theoj.org/papers/10.21105/joss.07471).
  [Version 0.11.3](https://pypi.org/project/ecologits/0.11.3/) and
  [estimator source](https://github.com/mlco2/ecologits/blob/main/ecologits/tracers/utils.py).
  Separate request estimates/statuses and the recorded electricity coefficient.
- NVIDIA: [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
  and [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
  Direct board-energy counters, power, memory, GPU identity and utilization.
- SURF: [accounting reference](https://servicedesk.surf.nl/wiki/plugins/viewsource/viewpagesrc.action?pageId=174457578).
  Source of the previously verified 192 SBU/H100-hour rate. It required login
  during the migration review, so the rate is not claimed to be newly verified.
- Slurm: [sbatch](https://slurm.schedmd.com/sbatch.html) and
  [GPU management](https://slurm.schedmd.com/gres.html#GPU_Management).
  Allocation limits, job IDs and GPU isolation.

## Research motivation and earlier multilingual design

- Ahia, O., et al. (2023).
  [Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models](https://aclanthology.org/2023.emnlp-main.614/).
  Motivates measuring equivalent-information token costs.
- Petrov, A., La Malfa, E., Torr, P., and Bibi, A. (2023).
  [Language Model Tokenizers Introduce Unfairness Between Languages](https://proceedings.neurips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html).
  Motivates token-length, latency and context-capacity comparisons.
- Buscemi, A., et al. (2025).
  [Strategic Communication and Language Bias in Multi-Agent LLM Coordination](https://arxiv.org/abs/2508.00032).
  Strategic-language motivation; no dialogue treatment is added here.
- Huynh, T.-K., et al. (2026).
  [Payoff scaling shapes cooperation in LLM agents across languages](https://arxiv.org/abs/2601.19082).
  Related work on language and incentives; native payoffs are unchanged.
- Kuang, P., et al. (2026).
  [BabelArena: A Large-Scale Multilingual Benchmark for LLM Agents](https://arxiv.org/abs/2609.23490)
  and [repository](https://github.com/QwenLM/BabelArena).
  Translation/verification guidance. BabelFlow itself is not integrated.
- Luccioni, A. S., Jernite, Y., and Strubell, E. (2024).
  [Power Hungry Processing: Watts Driving the Cost of AI Deployment?](https://doi.org/10.1145/3630106.3658542).
  Motivation for inference-energy measurements alongside outcomes.
- Kassis, T., et al. (2026).
  [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065).
  Earlier study acknowledgement for experimental-design/visualization workflows;
  not an agent algorithm or a runtime dependency.
- Mosquera, M., et al. (2024).
  [Can LLM-Augmented Autonomous Agents Cooperate?](https://arxiv.org/abs/2403.11381)
  and [CooperativeGPT](https://github.com/Cooperative-IA/CooperativeGPT).
  Compared as an alternative research implementation. Its Commons Harvest setup
  differs from the original Allelopathic Harvest design and is not adopted here.
- [Matplotlib constrained layout](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html).
  Applied to the previous study's figures. This repository does not copy its
  figure-generation scripts or simulated examples into the new results.

`measure.py` adapts the existing project's tested neutral `harvest/telemetry.py`:
the counter sampler is retained; request-file reading follows the HM logs;
CodeCarbon job tracking is restored. No KG code is reused. Historical provenance
is recorded in `history/extraction.json`; old outcomes are not new HM results.
