import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.autograd as autograd

from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import Adam

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
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_gp = 10


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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 2,
                      out_channels=ndf * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 4,
                      out_channels=ndf * 8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
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
generator.apply(init_weights)
discriminator.apply(init_weights)

# optimizer
g_optimizer = Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# train
batches_done = 0
num_epochs = 100
iters = 0
sample_interval = 400
# checking for sample generated image
fixed_noise = torch.rand(64, nz, 1, 1)
img_list = []

for epoch in range(num_epochs):
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
                  % (epoch + 1, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                  )
            if (iters % 500) == 0 or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                save_image(fake_x.data[:9], f"dataset/celeba/WGAN-GP_{epoch + 1}_{iters}.png", nrow=3, normalize=True)
        iters += 1
