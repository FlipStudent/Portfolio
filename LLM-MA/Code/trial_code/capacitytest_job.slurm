#!/bin/bash
#SBATCH --job-name=test_mistral
#SBATCH --time=01:00:00  # Max time (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G  # RAM
#SBATCH --partition=gpu  # Or your cluster’s partition
#SBATCH --gpus-per-node=a100:1  # Or your cluster’s partition

# Load modules if needed
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Run the Python script
# python test_compact.py
# python -m torch.distributed.run --nproc_per_node=2 test_iterative.py
python -m torch.distributed.run --nproc_per_node=1 capacitytest.py