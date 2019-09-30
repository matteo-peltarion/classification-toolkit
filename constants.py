import collections

# Template for message status
STATUS_MSG = "Batches done: {}/{} | Loss: {:04f} | Accuracy: {:04f}"

# Classes weights for loss
# akiec
# bcc
# bkl
# df
# mel
# nv
# vasc
CLASSES_WEIGHTS = collections.OrderedDict()
CLASSES_WEIGHTS['akiek'] = 10
CLASSES_WEIGHTS['bcc'] = 2
CLASSES_WEIGHTS['bkl'] = 1
CLASSES_WEIGHTS['df'] = 2
CLASSES_WEIGHTS['mel'] = 10
CLASSES_WEIGHTS['nv'] = 0.5
CLASSES_WEIGHTS['vasc'] = 2

# Mean and std computed on training set
# Mean: tensor([0.7633, 0.5459, 0.5704])
# Std: tensor([0.0898, 0.1184, 0.1331])
NORMALIZATION_MEAN = (0.7633, 0.5459, 0.5704)
NORMALIZATION_STD = (0.0898, 0.1184, 0.1331)
