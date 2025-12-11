#!/bin/bash
#SBATCH --job-name=test_mistral
#SBATCH --time=04:00:00  # Max time (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G  # RAM
#SBATCH --partition=gpu  # Or your cluster’s partition
#SBATCH --gpus-per-node=a100:1  # Or your cluster’s partition

# Load modules if needed
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Run the Python script
# python test_compact.py
python test.py "$@"
# python -m torch.distributed.run --nproc_per_node=2 test_iterative.py
#python -m torch.distributed.run --nproc_per_node=4 test_itercorrect.py