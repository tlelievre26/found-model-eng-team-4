#!/bin/bash
# Train (or resume) one run on an Insomnia A6000. Submit from the repo root:
#   sbatch scripts/slurm/train.sbatch configs/smoke.yaml [key=value ...]
#   sbatch scripts/slurm/train.sbatch configs/smoke.yaml run_name=smoke-r32 lora_rank=32
# Re-submitting the same command resumes the run from its resume state (a finished run refuses to retrain).
# Logs: $FME_PROJECT/fme_scratch_train/logs/ (override with sbatch -o). See docs/tuning_setup.md.
#SBATCH --job-name=fme-train
#SBATCH --account=pmg
#SBATCH --partition=pmg1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/insomnia001/depts/pmg/users/tjl2152/FME_project/fme_scratch_train/logs/%x-%j.out
set -euo pipefail

FME_PROJECT=${FME_PROJECT:-/insomnia001/depts/pmg/users/tjl2152/FME_project}
export FME_SHARED_DIR=${FME_SHARED_DIR:-$FME_PROJECT/fme_shared}
export HF_HOME=$FME_SHARED_DIR/models
export TMPDIR=$FME_PROJECT/fme_scratch_train/tmp   # /var/tmp is small and node-local
export WANDB_DIR=$FME_PROJECT/fme_scratch_train
mkdir -p "$TMPDIR"
source "$FME_PROJECT/envs/fme/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

config=$1
shift
# gpu_hourly_usd=0: Insomnia GPU time is not billed to the team, and L4 prices don't apply to an A6000.
# exec, so Slurm's TERM (sent 5 minutes before the time limit) reaches Python, which saves resume state.
exec python scripts/train.py --config "$config" gpu_hourly_usd=0 "$@"
