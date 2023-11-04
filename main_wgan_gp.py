import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.autograd as autograd

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
beta2 = 0.999
lambda_gp = 10

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


# make gradient-penalty(GP) for WGAN-GP
def compute_gradient_penalty(D: nn.Module, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    interpolate_samples = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True)  # 계산 그래프에 포함 & 역전파 허용
    interpolate_pred = D(interpolate_samples).view(batch_size, -1)
    fake_label = torch.ones(batch_size, 1).requires_grad_(False)

    gradients = autograd.grad(
        outputs=interpolate_pred,
        inputs=interpolate_samples,
        grad_outputs=fake_label,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean()  # (L2 norm -1)^2 의 mean => scalar
    return gradients_penalty


# define model & init params
generator = Generator()
discriminator = Discriminator()
generator.apply(weights_init)
discriminator.apply(weights_init)

# optimizer
g_optimizer = Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

batches_done = 0
n_epochs = 100
iters = 0
sample_interval = 400

for epoch in range(n_epochs):
    for i, (x, _) in enumerate(dataloader):
        # Train Discriminator
        d_optimizer.zero_grad()
        noise = torch.randn(len(x), nz, 1, 1)
        fake_x = generator(noise)

        real_pred = discriminator(x)
        fake_pred = discriminator(fake_x.detach().clone())

        gradient_penalty = compute_gradient_penalty(discriminator, x.data, fake_x.data)
        d_loss = -(torch.mean(real_pred) - torch.mean(fake_pred)) + lambda_gp * gradient_penalty
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        if i % 5 == 0:
            g_optimizer.zero_grad()
            fake_pred = discriminator(fake_x)
            g_loss = -torch.mean(fake_pred)
            g_loss.backward()
            g_optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G Loss: %f]"
                  % (epoch + 1, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                  )
            if batches_done % sample_interval == 0:
                vutils.save_image(fake_x.data[:25], f"dataset/celeba/WGAN-GP_{epoch}_{batches_done}.png", nrow=5,
                                  normalize=True)

            batches_done += 5