#!/bin/bash

# The network to use
NETWORK=SimpleCNN

# Optimizer
OPTIMIZER=Adam
#OPTIMIZER=SGD

# The learning rate
LR=0.001

BATCH_SIZE=8

NUM_EPOCHS=20

# Different levels for data augmentation
# 0: no DA
# 1: horizontal/vertical flips
# 2: random crops
# 3: color jitters
DATA_AUGMENTATION_LEVEL=2

# Either "--weighted-loss" or an empty string
WEIGHTED_LOSS="--weighted-loss"
#WEIGHTED_LOSS=""

# Either "--weighted-loss" or an empty string
NORMALIZE_INPUT="--normalize-input"
#NORMALIZE_INPUT=""

EXP_NAME="${NETWORK}_${OPTIMIZER}_${LR}_DA${DATA_AUGMENTATION_LEVEL}"
EXP_NAME="${EXP_NAME}${NORMALIZE_INPUT}${WEIGHTED_LOSS}"

# Just launch training with one single command
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

