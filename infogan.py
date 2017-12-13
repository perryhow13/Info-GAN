
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
gen_lr =  0.001
dis_lr =0.0002
beta1 = 0.5
beta2 = 0.999
LAMBDA =1.0

transform = transforms.Compose([transforms.Scale(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#test_set = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.fc0 = nn.Linear2D(NVEC_SIZE, 1024, bias=False)
    self.bn0 = nn.BatchNorm2d(1024)
    self.relu0 = nn.ReLU()
    self.fc1 = nn.Linear2D(1024, 128*7*7)
    self.bn1 = nn.BatchNorm2d(128)
    self.relu1 = nn.ReLU()
    self.convt0 = nn.ConvTranspose2d(128, 64, FILTER_SIZE, STRIDE, bias=False)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU()
    self.convt1 = nn.ConvTranspose2d(64, CHANNEL, FILTER_SIZE, STRIDE, bias=False)

  def weight_init(self, mean=0.0, std=0.02):
    for m in self._modules:
      if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Linear2D):
        self._modules[m].weight.data.normal_(mean, std)
        self._modules[m].bias.data.zero_()

  def forward(self, inputs):
    x = self.fc0(inputs)
    x = self.bn0(x)
    x = self.relu0(x)
    x = self.fc1(x)
    x = self.bn1(x.view(-1,128,7,7))
    x = self.relu1(x)
    x = self.convt0(x)
    x = self.bn2(x)
    x = self.relu2(x)
    output = self.convt1(x)
    return output

class QDiscriminator(nn.Module):
  def __init__(self):
    super(QDiscriminator, self).__init__()
    self.conv0 = nn.Conv2d(CHANNEL, 64, FILTER_SIZE, STRIDE, bias=False)
    self.lrelu0 =nn.LeakyReLU(negative_slope=0.1, inplace=True)

    self.conv1 = nn.Conv2d(64, 128, FILTER_SIZE, STRIDE, bias=False)
    self.bn0 = nn.BatchNorm2d(128)
    self.lreu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    self.fc0 = nn.Linear2D(,1024)
    self.bn1 = nn.BatchNorm2d(1024)

    self.fc1 = nn.Linear2D(1024, 10)

    self.fc2 = nn.Linear2D(1024, 128)
    self.lreu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    self.fc3 = nn.Linear2D(128, )

  def weight_init(self, mean=0.0, std=0.02):
    for m in self._modules:
      if isinstance(self._modules[m], nn.Conv2d) or isinstance(self._modules[m], nn.Linear2D):
        self._modules[m].weight.data.normal_(mean, std)
        self._modules[m].bias.data.zero_()

  def forward(self, input):
    output = self.model(input)
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
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=(beta1, beta2))
dis_optimizer = torch.optim.Adam(dis.parameters(), lr=dis_lr, betas=(beta1, beta2))

fixed_noise = Variable(torch.Tensor(BATCH_SIZE, 100, 1, 1).uniform_(-1, 1).cuda())

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
