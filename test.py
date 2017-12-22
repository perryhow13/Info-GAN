import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

BATCH_SIZE = 2
vector_size = 62
dis_c = Variable(torch.FloatTensor(BATCH_SIZE, 10))
con_c = Variable(torch.FloatTensor(BATCH_SIZE, 2))
noise = Variable(torch.FloatTensor(BATCH_SIZE, 62))



a, b = _noise_sample(dis_c, con_c, noise)

print a



