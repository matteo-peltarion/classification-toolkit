#!/bin/bash

# A tag for the experiment
TAG="res34_full1"

# The network to use
NETWORK=resnet34

# Optimizer
OPTIMIZER=Adam
#OPTIMIZER=SGD

# The starting learning rate
LR=0.005

BATCH_SIZE=8

NUM_EPOCHS=150

# Different levels for data augmentation
# 0: no DA
# 1: horizontal/vertical flips
# 2: random crops
# 3: color jitters
DATA_AUGMENTATION_LEVEL=3

# Either "--weighted-loss" or an empty string
WEIGHTED_LOSS="--weighted-loss"
#WEIGHTED_LOSS=""

# Either "--weighted-loss" or an empty string
NORMALIZE_INPUT="--normalize-input"
#NORMALIZE_INPUT=""

# Epochs after which to change (decrease) lr
# Either they're both set or both empty strings
MILESTONES_OPTION="--milestones"
MILESTONES="10 25 50 100"
#MILESTONES_OPTION=""
#MILESTONES=""

EXP_NAME="${NETWORK}_${OPTIMIZER}_${LR}_DA${DATA_AUGMENTATION_LEVEL}"
EXP_NAME="${TAG}_${EXP_NAME}${NORMALIZE_INPUT}${WEIGHTED_LOSS}"

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
    $MILESTONES_OPTION $MILESTONES \

