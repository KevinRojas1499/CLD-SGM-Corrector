#!/bin/bash

#SBATCH -J CLD
#SBATCH --account=gts-mtao8
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres=gpu:A100:1
#SBATCH -t8:00:00
#SBATCH -q inferno
#SBATCH -o training.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kevin.rojas@gatech.edu

cd /storage/home/hcoda1/7/krojas6/p-mtao8-0/repos/CLD-SGM-Corrector
module load anaconda3
module load gcc
conda deactivate
conda activate cld
bash scripts/train_mnist.sh