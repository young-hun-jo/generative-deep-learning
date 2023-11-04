import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop

from models.DCGAN import Generator, Discriminator, weights_init

# params
nz = 100
image_size = 64
batch_size = 128
lr = 0.0002
beta1 = 0.5

# load dataset
transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),
                         std=(0.5,0.5,0.5))
    ])

dataset = datasets.CelebA(root="dataset", split="train", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model
generator = Generator()
discriminator = Discriminator()

# init params in model
generator.apply(weights_init)
discriminator.apply(weights_init)

# loss
criterion = nn.BCELoss()

# checking for sample generated image
fixed_noise = torch.rand(64, nz, 1, 1)

# optimizer
optimizerD = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# train
img_list = []
g_losses = []
d_losses = []
iters = 0
num_epochs = 5

for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader, 0):
        # train Discriminator
        # 1. train based on real-image
        optimizerD.zero_grad()
        real_label = torch.ones(len(x))
        real_pred = discriminator(x).view(-1)
        # 2. train based on fake-image
        fake_label = torch.zeros(len(x))
        noise = torch.randn(len(x), nz, 1, 1)
        fake_x = generator(noise)
        fake_pred = discriminator(fake_x.detach()).view(-1)

        fake_loss = criterion(fake_pred, fake_label)
        real_loss = criterion(real_pred, real_label)
        d_loss = fake_loss + real_loss

        fake_loss.backward()
        real_loss.backward()
        optimizerD.step()

        # train Generator
        optimizerG.zero_grad()
        fake_pred = discriminator(fake_x).view(-1)
        g_loss = criterion(fake_pred, real_label)
        g_loss.backward()
        optimizerG.step()

        if i % 50 == 0:
            print("[%d|%d] [%d|%d]\tLoss D: %.4f\tLoss G: %.4f" % (epoch+1, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        d_losses.append(fake_loss.item() + real_loss.item())
        g_losses.append(g_loss.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1
