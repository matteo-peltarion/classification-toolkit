#!/bin/bash

# Set LR here
LR=0.01

python main.py \
    --num-epochs 100 \
    --batch-size 8 \
    --lr $LR \
    --network Alexnet \
    --exp-name alexnet && \
python main.py \
    --num-epochs 100 \
    --batch-size 8 \
    --lr $LR \
    --network VGG16  \
    --exp-name vgg16 && \
python main.py \
    --num-epochs 100 \
    --batch-size 8 \
    --lr $LR \
    --network resnet34  \
    --exp-name resnet34 && \
python main.py \
    --num-epochs 100 \
    --batch-size 8 \
    --lr $LR \
    --network resnet50  \
    --exp-name resnet50
