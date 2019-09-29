#!/bin/bash

# Just launch training with one single command
python main.py \
    --num-epochs 50 \
    --batch-size 8 \
    --network SimpleCNN \
    --lr 0.0001 \
    --resume
    #--network Alexnet
    #--network VGG16
