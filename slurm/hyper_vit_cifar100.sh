#!/bin/bash
#SBATCH --job-name=hyper_vit_cifar100
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/hyper_vit_cifar100_%j.out
#SBATCH --error=slurm/logs/hyper_vit_cifar100_%j.err

# Experiment: Hypernetwork-conditioned QKV on Split-CIFAR-100

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "========================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Change to project directory
cd ${SLURM_SUBMIT_DIR}

# Create directories
mkdir -p slurm/logs
mkdir -p results

# Run experiment
echo ""
echo "Experiment: Hypernetwork-conditioned QKV (input-conditioned)"
echo ""

uv run python -c "
from hyper_vit_cifar100 import main
main()
"

uv run python -c "
from aggregate_results import main
main()
"

exit_code=$?

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "Exit code: $exit_code"
echo "========================================"

exit $exit_code
