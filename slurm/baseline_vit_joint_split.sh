#!/bin/bash
#SBATCH --job-name=baseline_vit_joint
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/baseline_vit_joint_%j.out
#SBATCH --error=slurm/logs/baseline_vit_joint_%j.err

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
echo "Baseline: ViT trained on union of Split-CIFAR-100 tasks"
echo ""

uv run python -c "
from baseline_vit_joint_split import main
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
