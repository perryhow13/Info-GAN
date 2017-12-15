
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils


NVEC_SIZE = 74
FILTER_SIZE = 4
STRIDE = 2
CHANNEL = 1
BATCH_SIZE = 50
EPOCH = 10
GEN_LR =  0.001
DIS_LR =0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA =1.0
POOLING = 1

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#test_set = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(nn.Conv2d(CHANNEL, 64, FILTER_SIZE, STRIDE, POOLING, bais=False),
                              nn.LeakyReLU(0.1),
                              nn.Conv2d(64, 128, FILTER_SIZE, STRIDE, POOLING, bias=False),
                              nn.BatchNorm2d(128),
                              nn.LeakyReLU(0.1),
                              nn.Conv2d(128, 1024, FILTER_SIZE+3, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.LeakyReLU(0.1),
                              nn.Conv2d(1024, CHANNEL, 1, bais=False),
                              nn.Sigmoid())

  def weight_init(self, mean=0.0, std=0.02):
    for m in self._modules:
      if isinstance(self._modules[m], nn.Conv2d):
        self._modules[m].weight.data.normal_(mean, std)
        self._modules[m].bias.data.zero_()

  def forward(self,input):
    output = self.model(input).squeeze()
    return output

class Q(nn.Module):
  def __init__(self):
    self.conv0 = nn.Conv2d(CHANNEL, 64, FILTER_SIZE, STRIDE, POOLING, bias=False)
    self.lr0 = nn.LeakyReLU(0.1)
    self.conv1 = nn.Conv2d(64, 128, FILTER_SIZE, STRIDE, POOLING, bias=False)
    self.bn1 = nn.BatchNorm2d(128)
    self.lr1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(128, 1024, FILTER_SIZE+3, STRIDE-1, bias=False)
    self.bn2 = nn.BatchNorm2d(1024)
    self.lr2 = nn.LeakyReLU(0.1)
    self.conv3 =nn.Conv2d(1024, 128, 1, bias=False)
    self.bn3 = nn.BatchNorm2d(128)
    self.lr3 = nn.LeakyReLU(0.1)
    self.discrete = nn.Conv2d(128, 10, 1, bias=False)
    self.mean = nn.Conv2d(128, 2, 1, bias=False)
    self.std = nn.Conv2d(128, 2, 1, bias=False)

  def weight_init(self, mean=0.0, std=0.02):
  for m in self._modules:
    if isinstance(self._modules[m], nn.Conv2d):
      self._modules[m].weight.data.normal_(mean, std)
      self._modules[m].bias.data.zero_()

  def forward(self, inputs):
    x = self.conv0(inputs)
    x = self.lr0(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.lr1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.lr2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.lr3(x)

    disc = self.discrete(x).squeeze()
    mean = self.mean(x).squeeze()
    std = self.std(x).squeeze().exp()

    return (disc, mean, std)

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.model = nn.Sequential(nn.ConvTranspose2d(NVEC_SIZE, 1024, FILTER_SIZE-3, STRIDE-1, bias=False),
                               nn.BatchNorm2d(1024),
                               nn.ReLU(),
                               nn.ConvTranspose2d(1024, 128, FILTER_SIZE+3, STRIDE-1, bias=False),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.ConvTranspose2d(128, 64, FILTER_SIZE, STRIDE, POOLING, bias=False),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.ConvTranspose2d(64, CHANNEL, FILTER_SIZE, STRIDE, POOLING, bias=False),
                               nn.Sigmoid())

  def weight_init(self, mean=0.0, std=0.02):
    for m in self._modules:
      if isinstance(self._modules[m], nn.ConvTranspose2d):
        self._modules[m].weight.data.normal_(mean, std)
        self._modules[m].bias.data.zero_()

  def forward(self, inputs):
    output = self.model(inputs)
    return output


real_label = Variable(torch.ones(BATCH_SIZE)).cuda()
fake_label = Variable(torch.zeros(BATCH_SIZE).cuda())

gen = Generator()
gen.weight_init()
gen.cuda()

dis = QDiscriminator()
dis.weight_init()
dis.cuda()

loss_func = nn.BCELoss()
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=GEN_LR, betas=(BETA1, BETA2))
dis_optimizer = torch.optim.Adam(dis.parameters(), lr=DIS_LR, betas=(BETA1, BETA2))

fixed_noise = Variable(torch.Tensor(BATCH_SIZE, NVEC_SIZE-12, 1, 1).uniform_(-1, 1).cuda())

for epoch in range(EPOCH):
  for i, data in enumerate(train_loader):
    dis.zero_grad()
    real_data, _ = data
    real_data = Variable(real_data.cuda())
    real_output = dis(real_data)
    real_loss = loss_func(real_output.squeeze(), real_label)
    real_loss.backward()
    dis_optimizer.step()

    dis.zero_grad()
    noise = Variable(torch.Tensor(BATCH_SIZE, 100, 1, 1).uniform_(-1,1).cuda())
    gen_data = gen(noise)
    fake_output = dis(gen_data)
    fake_loss = loss_func(fake_output.squeeze(), fake_label)
    fake_loss.backward()
    #total_loss = real_loss + fake_loss
    #total_loss.backward()
    dis_optimizer.step()
    print "Total Loss: " + str(fake_loss.data)

    gen.zero_grad()
    new_gen_data = gen(noise)
    output = dis(new_gen_data)
    loss = loss_func(output.squeeze(), real_label)
    loss.backward()
    gen_optimizer.step()
    print "Loss when input = fake image but with real label:" + str(loss.data)
    print "--------------------------------------------------------------------"
    if (i+1) % 100 == 0:
      print "Step: " + str(i+1)
      gen_img = gen(fixed_noise)
      vutils.save_image(gen_img.data, 'generated_img_in_epoch_%s_step_%s.png' % (str(epoch),str(i+1)), normalize=True)



"""
<Significant founds>

Noise must be random uniform in interval [-1, 1], otherwise it just does not get trained
"""

"""
<Things to try>
Uncomment the comments in the training loop and comment "total_loss = real_loss + fake_loss"
and "total_loss.backward()"
"""
