import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp

seq2 = np.load(
    "datasets" + os.sep + "kinetic_features" + os.sep + "summe" + os.sep + "FLOW" + os.sep + "features" + os.sep  + "Air_Force_One.npy")  # you can provide features from another video
seq3 = np.load(
    "datasets" + os.sep + "kinetic_features" + os.sep + "summe" + os.sep + "RGB" + os.sep + "features" + os.sep + "Air_Force_One.npy")

comparison = seq2 == seq3
equal_arrays = comparison.all()

print(equal_arrays)