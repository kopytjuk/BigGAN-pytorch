#!/bin/bash
# 1. Updates GAN code
# 2. Starts a training/experiment
# 3. Starts tensorboard to monitor training progress

dt=$(date '+%Y%m%d-%H%M%S')
experimentsfolder="/home/mosaic-trainer/experiments/${dt}_bigGAN"
datasetfolder="/home/mosaic-trainer/data/cub-200-cropped"

python main.py \
    --batch_size 20 \
    --imsize 128 \
    --dataset off \
    --adv_loss hinge \
    --version alpha_v1 \
    --use_tensorboard True \
    --experiment_path "$experimentsfolder" \
    --image_path "$datasetfolder" \
    --model_save_step 1 \
    --sample_step 100 \
    & \
tensorboard --logdir="$experimentsfolder/logs/tf_logs" --host 0.0.0.0 --port 6006
