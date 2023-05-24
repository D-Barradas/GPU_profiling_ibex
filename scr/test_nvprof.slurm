#!/bin/bash -l 
#SABTCH --job-name=nvprof
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100,gpu_ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load dl
module load cuda/10.2.89
module load pytorch/1.5.1 torchvision/0.6.1
module list

export DATA_DIR="/ibex/scratch/shaima0d/ML_framework_testing/pytorch_examples/imagenet_dataset/tiny-imagenet-200"
batch_size=256
epochs=1
workers=8

cmd="python3 ./train.py -a resnet50 --epochs ${epochs} --batch-size ${batch_size}  --workers ${workers} ${DATA_DIR}"
nvprof -o profile.nvvp ${cmd}
