import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms

from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop

# load dataset
image_size = 64
batch_size = 128

transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])

dataset = datasets.CelebA(root="dataset", split="train", transform=transforms, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

nz = 100
ngf = 64
ndf = 64
nc = 3
lr = 0.00005
beta1 = 0.5


def init_weights(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.lower().find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.lower().find('batchnorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz,
                               out_channels=ngf * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 8,
                               out_channels=ngf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 4,
                               out_channels=ngf * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 2,
                               out_channels=ngf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf,
                               out_channels=nc,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nc,
                      out_channels=ndf,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf,
                      out_channels=ndf * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 2,
                      out_channels=ndf * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 4,
                      out_channels=ndf * 8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 8,
                      out_channels=1,
                      kernel_size=4,
                      stride=2,
                      padding=0,
                      bias=False)
        )

    def forward(self, x):
        x = self.main(x)
        return x


# checking for sample generated image
fixed_noise = torch.rand(64, nz, 1, 1)

# define model && init params
generataor = Generator()
discriminator = Discriminator()
generataor.apply(init_weights)
discriminator.apply(init_weights)

# optimizer
g_optimizer = RMSprop(generataor.parameters(), lr=lr)
d_optimizer = RMSprop(discriminator.parameters(), lr=lr)

# train
img_list = []
g_losses = []
d_losses = []
iters = 0
num_epochs = 50

for epoch in range(num_epochs):
    for i, (x, _) in enumerate(dataloader):
        # train Discriminator
        d_optimizer.zero_grad()
        noise = torch.randn(len(x), nz, 1, 1)
        fake_x = generataor(noise)

        real_pred = discriminator(x).view(-1)
        fake_pred = discriminator(fake_x.detach().clone()).view(-1)

        d_loss = -(torch.mean(real_pred) - torch.mean(fake_pred))
        d_loss.backward()
        d_optimizer.step()

        # weight cliping for Discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        # train Generator
        if i % 5 == 0:
            g_optimizer.zero_grad()
            fake_x = generataor(noise)
            fake_pred = discriminator(fake_x).view(-1)
            g_loss = -torch.mean(fake_pred)
            g_loss.backward()
            g_optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G Loss: %f"
                  % (epoch + 1, num_epochs, iters, len(dataloader), d_loss.item(), g_loss.item())
                  )
        if (iters % 500) == 0 or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            save_image(fake_x.data[:9], f"dataset/celeba/WGAN_{epoch + 1}_{iters}.png", nrow=3, normalize=True)
            print("Saved image!")
        iters += 1