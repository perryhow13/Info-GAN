import torch
a = torch.Tensor([-0.1, 0.2, -3, 4])
b = torch.nn.ReLU()
c = b(a)

print a

relu1 = torch.nn.ReLU(inplace=True)

d = relu1(torch.autograd.Variable(a))
