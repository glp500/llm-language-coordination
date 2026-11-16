#!/bin/bash
# Pass the plan's exact wall time to sbatch; one H100 and one resident model per block.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
root="${1:?Provide a frozen allocation directory}"
index="${2:-0}"
module load 2025
module load Miniforge3/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda/runtime"
# Module reloads can leave base Python first on PATH even when this prefix is active.
export COORDINATION_RUN_ROOT="$root"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export COORDINATION_ALLOCATED_SECONDS="$(squeue -h -j "$SLURM_JOB_ID" -o '%l' | "$CONDA_PREFIX/bin/python" -c '
import sys
value=sys.stdin.read().strip()
days,value=value.split("-",1) if "-" in value else ("0",value)
print(int(days)*86400+sum(int(v)*60**i for i,v in enumerate(reversed(value.split(":")))))
')"
# TERM arrives before Slurm's hard kill so evidence and energy records can flush.
limit=$((COORDINATION_ALLOCATED_SECONDS - 45))
test "$limit" -gt 0
"$CONDA_PREFIX/bin/python" measure.py "$root/energy/$(printf '%04d' "$index")" -- \
    timeout --signal=TERM --kill-after=20s "${limit}s" \
    "$CONDA_PREFIX/bin/python" run.py batch "$root" --index "$index"
