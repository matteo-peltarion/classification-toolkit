#!/bin/bash

# The network to use
NETWORK=resnet34

# Optimizer
OPTIMIZER=Adam
#OPTIMIZER=SGD

# The learning rate
LR=0.001

BATCH_SIZE=8

NUM_EPOCHS=30

# Different levels for data augmentation
# 0: no DA
# 1: horizontal/vertical flips
# 2:
DATA_AUGMENTATION_LEVEL=2

# Either "--weighted-loss" or an empty string
WEIGHTED_LOSS="--weighted-loss"
#WEIGHTED_LOSS=""

# Either "--weighted-loss" or an empty string
NORMALIZE_INPUT="--normalize-input"
#NORMALIZE_INPUT=""

for LR in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001; do

    EXP_NAME="${NETWORK}_${OPTIMIZER}_${LR}_DA${DATA_AUGMENTATION_LEVEL}"
    EXP_NAME="${EXP_NAME}${NORMALIZE_INPUT}${WEIGHTED_LOSS}"

    python main.py \
        --network $NETWORK \
        --num-epochs $NUM_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --optimizer $OPTIMIZER \
        --data-augmentation-level $DATA_AUGMENTATION_LEVEL \
        --exp-name $EXP_NAME \
        $WEIGHTED_LOSS \
        $NORMALIZE_INPUT \

done

