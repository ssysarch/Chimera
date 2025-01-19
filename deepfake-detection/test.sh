#!/bin/bash

ARCH="CLIP:ViT-L/14"
CKPT="pretrained_weights/fc_weights.pth"
RESULT_FOLDER="results/univ"

BATCH_SIZE=16

python3 validate.py --arch=$ARCH --ckpt=$CKPT --result_folder=$RESULT_FOLDER --batch_size=$BATCH_SIZE



ARCH="FAT:ViT-L/14"
CKPT="pretrained_weights/fatformer_4class_ckpt.pth"
RESULT_FOLDER="results/fat"

python3 validate.py --arch=$ARCH --ckpt=$CKPT --result_folder=$RESULT_FOLDER --batch_size=$BATCH_SIZE

