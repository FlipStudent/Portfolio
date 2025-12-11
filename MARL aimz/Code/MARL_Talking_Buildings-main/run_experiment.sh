#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=experiment
#SBATCH --output=experiment.out

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
pip install pandas
module load typing-extensions/4.10.0-GCCcore-13.2.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load matplotlib/3.5.2-foss-2022a

python src/generate_results.py experiment final1
