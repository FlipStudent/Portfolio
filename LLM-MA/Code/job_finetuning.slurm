#!/bin/bash
#SBATCH --job-name=mistral_lora_train
#SBATCH --time=03:00:00  # Max time (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G  # RAM
#SBATCH --partition=gpu  # Or your cluster’s partition
#SBATCH --gpus-per-node=a100:1  # Or your cluster’s partition


# Load modules if needed
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Run the Python script
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune.py "$@"
#python -m torch.distributed.run finetune.py "$@"
#python -m torch.distributed.run --nproc_per_node=2 finetune.py