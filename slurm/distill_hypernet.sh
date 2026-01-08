#!/bin/bash
#SBATCH --job-name=distill_hypernet
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/distill_hypernet_%j.out
#SBATCH --error=slurm/logs/distill_hypernet_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "========================================"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

cd ${SLURM_SUBMIT_DIR}
mkdir -p slurm/logs
mkdir -p results

echo ""
echo "Stage 2: Distill hypernetwork from saved LoRA weights"
echo ""

uv run python -c "
from distill_hypernet import main
main()
"

exit_code=$?

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "Exit code: $exit_code"
echo "========================================"

exit $exit_code
