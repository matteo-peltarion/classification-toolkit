#!/bin/bash

# Just launch training with one single command
python main.py \
    --num-epochs 50 \
    --batch-size 8 \
    --lr 0.001 \
    --network Alexnet
    #--network SimpleCNN
    #--resume
    #--network VGG16
