import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math 
import itertools
from glob import glob