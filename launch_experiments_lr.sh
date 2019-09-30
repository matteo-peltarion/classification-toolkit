#!/bin/bash

# Set LR here
LR=0.01

#for LR in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001; do
    #echo $LR
#done

#exit 1

#for LR in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001; do
#for LR in 0.01; do
for LR in 0.005 0.001 0.0005 0.0001 0.00005 0.00001; do
    python main.py \
        --num-epochs 100 \
        --batch-size 8 \
        --lr $LR \
        --network SimpleCNN \
        --exp-name "SimpleCNN_LR_${LR}"
done

