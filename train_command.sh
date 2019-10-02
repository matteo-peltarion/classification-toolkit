#!/bin/bash

# A custom tag for the experiment
EXP_TAG="run6_"

# The network to use
#NETWORK=Alexnet
#NETWORK=resnet34
#NETWORK=resnet50
#NETWORK=resnet101
NETWORK=resnet152

# Optimizer
OPTIMIZER=Adam
#OPTIMIZER=SGD

# The starting learning rate
LR=0.01

#BATCH_SIZE=8 # ok for resnet50
BATCH_SIZE=6 # ok for resnet152

NUM_EPOCHS=600

# Different levels for data augmentation
# 0: no DA
# 1: horizontal/vertical flips
# 2: random crops
# 3: color jitters
DATA_AUGMENTATION_LEVEL=3

# Experiment specific weights
#CLASSES_WEIGHTS['akiec'] = 10
#CLASSES_WEIGHTS['bcc'] = 3
#CLASSES_WEIGHTS['bkl'] = 2
#CLASSES_WEIGHTS['df'] = 2
#CLASSES_WEIGHTS['mel'] = 10
#CLASSES_WEIGHTS['nv'] = 2
#CLASSES_WEIGHTS['vasc'] = 2

# Weights for cross entropy loss
# Either they're both set or both empty strings
CLASS_WEIGHTS_OPTIONS="--class-weights"
#CLASS_WEIGHTS="5 3 2 2 5 2 2"
CLASS_WEIGHTS="2.5 2.5 2.5 2.5 2 1 2"
#CLASS_WEIGHTS_OPTIONS=""
#CLASS_WEIGHTS=""

if [ -n "$CLASS_WEIGHTS_OPTIONS" ]; then
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
MILESTONES="25 250 400"
#MILESTONES_OPTION=""
#MILESTONES=""

PRETRAINED_OPTION="--use-pretrained"

if [ -n "$PRETRAINED_OPTION" ]; then
    PRETRAINED_TAG="_pretrained"
fi

# Weight decay option
WEIGHT_DECAY_OPTION="--weight-decay"
WEIGHT_DECAY="0.00001"
#WEIGHT_DECAY_OPTION=""
#WEIGHT_DECAY=""

if [ -n "$WEIGHT_DECAY_OPTION" ]; then
    WEIGHT_DECAY_TAG="_WD${WEIGHT_DECAY}"
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
    $NORMALIZE_INPUT_OPTION \
    $PRETRAINED_OPTION \
    $MILESTONES_OPTION $MILESTONES \
    $WEIGHT_DECAY_OPTION $WEIGHT_DECAY \
    $CLASS_WEIGHTS_OPTIONS $CLASS_WEIGHTS \

