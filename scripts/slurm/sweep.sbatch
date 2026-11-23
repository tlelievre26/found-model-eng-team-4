#!/bin/bash
# Run the hyperparameter grid (baseline eval, then every trial in order) on one Insomnia A6000.
# Submit from the repo root:
#   sbatch scripts/slurm/sweep.sbatch                      # configs/sweep.yaml
#   sbatch scripts/slurm/sweep.sbatch path/to/sweep.yaml [key=value ...]
# Re-submitting skips finished trials and resumes a partial one. Logs: $FME_PROJECT/fme_scratch_train/logs/.
#SBATCH --job-name=fme-sweep
#SBATCH --account=pmg
#SBATCH --partition=pmg1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/insomnia001/depts/pmg/users/tjl2152/FME_project/fme_scratch_train/logs/%x-%j.out
set -euo pipefail

FME_PROJECT=${FME_PROJECT:-/insomnia001/depts/pmg/users/tjl2152/FME_project}
export FME_SHARED_DIR=${FME_SHARED_DIR:-$FME_PROJECT/fme_shared}
export HF_HOME=$FME_SHARED_DIR/models
export TMPDIR=$FME_PROJECT/fme_scratch_train/tmp
export WANDB_DIR=$FME_PROJECT/fme_scratch_train
mkdir -p "$TMPDIR"
source "$FME_PROJECT/envs/fme/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

sweep=${1:-configs/sweep.yaml}
shift || true
exec python scripts/run_sweep.py --sweep "$sweep" gpu_hourly_usd=0 "$@"
