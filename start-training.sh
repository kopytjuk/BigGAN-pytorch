#!/bin/bash
# 1. Updates GAN code
# 2. Starts a training/experiment
# 3. Starts tensorboard to monitor training progress

dt=$(date '+%Y%m%d-%H%M%S')
experimentsfolder="/home/mosaic-trainer/experiments/$dt"
datasetfolder="/home/mosaic-trainer/data/cub-200-cropped/images"

python main.py \
    --batch_size 64 \
    --dataset off \
    --adv_loss hinge \
    --version alpha_v1 \
    --use_tensorboard True \
    --experiment_path "$experimentsfolder"
    --image_path "$dataset_folder" \
    & \
tensorboard --logdir="$experimentsfolder/logs" --host 0.0.0.0 --port 6006