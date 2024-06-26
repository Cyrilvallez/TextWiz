#!/bin/bash

#SBATCH --job-name=memory_estimator
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=200G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:6
#SBATCH --chdir=/cluster/raid/home/vacy/TextWiz

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate textwiz

# srun -u --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 --mem=5G python3 -u memory_estimator_wrapper.py "$@"

python3 -u memory_estimator_wrapper.py "$@"

conda deactivate
