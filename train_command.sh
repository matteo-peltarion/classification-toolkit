#!/bin/bash

# A custom tag for the experiment
EXP_TAG="run2_"

# The network to use
#NETWORK=Alexnet
#NETWORK=resnet34
NETWORK=resnet50
#NETWORK=resnet101
#NETWORK=resnet152

# Optimizer
OPTIMIZER=Adam
#OPTIMIZER=SGD

# The starting learning rate
LR=0.001

BATCH_SIZE=8 # ok for resnet50

#BATCH_SIZE=6 # ok for resnet152

NUM_EPOCHS=300

# Different levels for data augmentation
# 0: no DA
# 1: horizontal/vertical flips
# 2: random crops
# 3: color jitters
DATA_AUGMENTATION_LEVEL=3

# Either "--weighted-loss" or an empty string
WEIGHTED_LOSS_OPTION="--weighted-loss"
#WEIGHTED_LOSS_OPTION=""

if [ -n "$WEIGHTED_LOSS_OPTION" ]; then
    WEIGHTED_LOSS_TAG="_weighted_loss"
fi

# Either "--weighted-loss" or an empty string
NORMALIZE_INPUT_OPTION="--normalize-input"
#NORMALIZE_INPUT_OPTION=""

if [ -n "$NORMALIZE_INPUT_OPTION" ]; then
    NORMALIZE_INPUT_TAG="_normalize_input"
fi

# Epochs after which to change (decrease) lr
# Either they're both set or both empty strings
MILESTONES_OPTION="--milestones"
MILESTONES="25 50 100 200"
#MILESTONES_OPTION=""
#MILESTONES=""

PRETRAINED_OPTION="--use-pretrained"

if [ -n "$PRETRAINED_OPTION" ]; then
    PRETRAINED_TAG="_pretrained"
fi

# Build the name of the experiments from configuration
EXP_NAME="${EXP_TAG}${NETWORK}${PRETRAINED_TAG}_${OPTIMIZER}_${LR}_DA${DATA_AUGMENTATION_LEVEL}"
EXP_NAME="${EXP_NAME}${NORMALIZE_INPUT_TAG}${WEIGHTED_LOSS_TAG}"

# Just launch training with one single command
python main.py \
    --network $NETWORK \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --data-augmentation-level $DATA_AUGMENTATION_LEVEL \
    --exp-name $EXP_NAME \
    $WEIGHTED_LOSS_OPTION \
    $NORMALIZE_INPUT_OPTION \
    $PRETRAINED_OPTION \
    $MILESTONES_OPTION $MILESTONES \

