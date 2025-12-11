#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=pre-iam
#SBATCH --output=pre-iam.out

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
pip install scikit-image
pip install transformers
pip install evaluate
pip install jiwer
module load OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib
module load tqdm/4.64.1-GCCcore-12.2.0
module load Pillow/10.0.0-GCCcore-12.3.0
module load torchvision/0.13.1-foss-2022a
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load matplotlib/3.5.2-foss-2022a

python pretrain.py
