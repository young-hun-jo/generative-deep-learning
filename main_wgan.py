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
lr = 0.00005
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

# checking for sample generated image
fixed_noise = torch.rand(64, nz, 1, 1)

# define model && init params
generator = Generator()
discriminator = Discriminator()
generator.apply(weights_init)
discriminator.apply(weights_init)

# optimizer
g_optimizer = RMSprop(generator.parameters(), lr=lr)
d_optimizer = RMSprop(discriminator.parameters(), lr=lr)

img_list = []
g_losses = []
d_losses = []
iters = 0
num_epochs = 100

for epoch in range(num_epochs):
    for i, (x, _) in enumerate(dataloader):
        # train Discriminator
        d_optimizer.zero_grad()
        noise = torch.randn(len(x), nz, 1, 1)
        fake_x = generator(noise)

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
            fake_x = generator(noise)
            fake_pred = discriminator(fake_x).view(-1)
            g_loss = -torch.mean(fake_pred)
            g_loss.backward()
            g_optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G Loss: %f"
                  % (epoch + 1, num_epochs, iters, len(dataloader), d_loss.item(), g_loss.item())
                  )
        if (iters % 500) == 0 or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():  # not caching middle-outputs because of no-backpropagation
                fake_img = generator(fixed_noise).detach()
            img_list.append(vutils.make_grid(fake_img, padding=2, normalize=True))
        iters += 1
