#!/bin/bash
#SBATCH --job-name=train_lora_save
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/train_lora_save_%j.out
#SBATCH --error=slurm/logs/train_lora_save_%j.err

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
mkdir -p results/lora

echo ""
echo "Stage 1: Train per-task LoRA and save CLS samples"
echo ""

uv run python -c "
from train_lora_save import main
main()
"

exit_code=$?

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "Exit code: $exit_code"
echo "========================================"

exit $exit_code
