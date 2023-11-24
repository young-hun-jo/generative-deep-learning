import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from typing import Tuple
from itertools import chain

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.io import read_image
from torchvision.utils import save_image, make_grid
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim import lr_scheduler


class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        self.init_filter = 3
        self.gen_n_filter = 32

        self.d1 = self.create_downsample_layer(self.init_filter, self.gen_n_filter)
        self.d2 = self.create_downsample_layer(self.gen_n_filter, self.gen_n_filter * 2)
        self.d3 = self.create_downsample_layer(self.gen_n_filter * 2, self.gen_n_filter * 4)
        self.d4 = self.create_downsample_layer(self.gen_n_filter * 4, self.gen_n_filter * 8)

        self.u1 = self.create_upsample_layer(self.gen_n_filter * 8, self.gen_n_filter * 4)
        self.u2 = self.create_upsample_layer(self.gen_n_filter * 4, self.gen_n_filter * 2)
        self.u3 = self.create_upsample_layer(self.gen_n_filter * 2, self.gen_n_filter)
        self.u4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.gen_n_filter,
                      out_channels=self.init_filter,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

    def create_downsample_layer(self, in_filter: int, out_filter: int):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_filter,
                      out_channels=out_filter,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.InstanceNorm2d(num_features=out_filter),
            nn.ReLU()
        )
        return layer

    def create_upsample_layer(self, in_filter: int, out_filter: int):
        layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_filter,
                      out_channels=out_filter,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.InstanceNorm2d(num_features=out_filter),
            nn.ReLU()
        )
        return layer

    def skip_connection(self, x1, x2):
        return torch.cat([x1, x2], dim=1)

    def forward(self, x):
        x = self.d1(x)
        self.d1_output = x

        x = self.d2(x)
        self.d2_output = x

        x = self.d3(x)
        self.d3_output = x

        x = self.d4(x)
        self.d4_output = x

        x = self.u1(x)
        x = x + self.d3_output  # +=: inplace operation error occured("RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:")

        x = self.u2(x)
        x = x + self.d2_output

        x = self.u3(x)
        x = x + self.d1_output

        x = self.u4(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(num_features=in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(num_features=in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, num_residual_blocks: int):
        super(GeneratorResNet, self).__init__()

        channels = 3
        out_features = 64

        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(padding=channels),
            nn.Conv2d(in_channels=channels, out_channels=out_features, kernel_size=7),
            nn.InstanceNorm2d(num_features=out_features),
            nn.ReLU(inplace=True)
        )

        # down-sampling
        in_features = out_features
        self.d1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=in_features * 2),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features * 2, out_channels=in_features * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=in_features * 4),
            nn.ReLU(inplace=True)
        )

        # residual-blocks
        self.num_residual_blocks = num_residual_blocks
        for i in range(num_residual_blocks):
            layer = ResidualBlock(in_features=in_features * 4)
            setattr(self, f'b{i + 1}', layer)

        # up-sampling
        self.u1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_features * 4, out_channels=in_features * 4 // 2, kernel_size=3, stride=1,
                      padding=1),
            nn.InstanceNorm2d(num_features=in_features * 4 // 2),
            nn.ReLU(inplace=True)
        )
        self.u2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_features * 4 // 2, out_channels=in_features * 4 // 2 // 2, kernel_size=3, stride=1,
                      padding=1),
            nn.InstanceNorm2d(num_features=in_features * 4 // 2 // 2),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=in_features * 4 // 2 // 2, out_channels=channels, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)

        # down-sample
        x = self.d1(x)
        x = self.d2(x)

        # residual-block
        for i in range(self.num_residual_blocks):
            l = getattr(self, f'b{i + 1}')
            x = l(x)
            verbose_shape(x)

        # up-sample
        x = self.u1(x)
        x = self.u2(x)

        y = self.layer2(x)

        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.init_filter = 3
        self.disc_n_filter = 32

        self.conv1 = self.create_conv_layer(self.init_filter, self.disc_n_filter, stride=2, padding=1)
        self.conv2 = self.create_conv_layer(self.disc_n_filter, self.disc_n_filter * 2, stride=2, padding=1)
        self.conv3 = self.create_conv_layer(self.disc_n_filter * 2, self.disc_n_filter * 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.disc_n_filter * 4, out_channels=self.disc_n_filter * 8, kernel_size=4,
                               padding='same')
        self.final_conv = nn.Conv2d(in_channels=self.disc_n_filter * 8, out_channels=1, kernel_size=4, stride=2)
        self.output_shape = (1, 15, 15)  # C, H, W

    def create_conv_layer(self, in_filter: int, out_filter: int, stride: int, padding: int):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_filter,
                      out_channels=out_filter,
                      kernel_size=4,
                      stride=stride,
                      padding=padding),
            nn.InstanceNorm2d(num_features=out_filter),
            nn.LeakyReLU(negative_slope=0.2)
        )
        return layer

    def forward(self, x):
        verbose_shape(x)
        x = self.conv1(x)
        verbose_shape(x)
        x = self.conv2(x)
        verbose_shape(x)
        x = self.conv3(x)
        verbose_shape(x)
        x = self.conv4(x)
        verbose_shape(x)
        y = self.final_conv(x)
        verbose_shape(y)

        return y


def verbose_shape(x):
    pass
    # print(x.shape)


def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ReplayBuffer(object):
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:  # 배치 사이즈 내 single 데이터 loop
            element = torch.unsqueeze(element, 0)  # 0번째에 1개 축 추가
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  # 50%
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class OrangeAppelDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        self.transforms = transforms

        if is_train:
            self.path_A = "./dataset/trainA"
            self.path_B = "./dataset/trainB"
            self.type = "train"
        else:
            self.path_A = "./dataset/testA"
            self.path_B = "./dataset/testB"
            self.type = "test"

        self.A_files = os.listdir(self.path_A)
        self.B_files = os.listdir(self.path_B)

        self.A = [f'./dataset/{self.type}A/{f}' for f in self.A_files]
        self.B = [f'./dataset/{self.type}B/{f}' for f in self.B_files]

    def __len__(self):
        return max(len(self.A), len(self.B))

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        a_idx = idx % len(self.A)
        b_idx = idx % len(self.B)

        a_img = read_image(self.A[a_idx]).float()
        b_img = read_image(self.B[b_idx]).float()

        if self.transforms:
            a_img = self.transforms(a_img)
            b_img = self.transforms(b_img)
        return a_img, b_img


def apply_multi_gpu_to_model(*models, gpu_list: str, is_multi_gpu=False):
    if is_multi_gpu:
        # DataParallel 씌우면 빌드한 모델의 custom init vars 날라감..
        models = [DataParallel(m, device_ids=gpu_list).cuda() for m in models]
        return (*models,)
    else:
        models = [m.cuda() for m in models]
        return (*models,)


def sample_images(batches_done, generator_AB: nn.Module, generator_BA: nn.Module):
    real_A, real_B = next(iter(valid_loader))

    generator_AB.eval()
    generator_BA.eval()

    fake_A = generator_BA(real_B).cpu()
    fake_B = generator_AB(real_A).cpu()

    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    image_grid = torch.cat((real_A, real_B, fake_A, fake_B), 1)
    save_image(image_grid, f"./dataset/samples/{batches_done}.png", normalize=True)
    print("Save image!")


# image Resize를 하기 위해 interpolation을 수행하는데, 그 때 사용하는 방법 중 하나로 PIL의 bicubic 방법을 이용
# 각 채널별 (x - x_mean) / x_std
transforms = transforms.Compose([
    transforms.Resize(int(256 * 1.12), Image.BICUBIC, antialias=True),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)

train_dataset = OrangeAppelDataset(is_train=True, transforms=transforms)
valid_dataset = OrangeAppelDataset(is_train=False, transforms=transforms)

train_loader = DataLoader(train_dataset, batch_size=10)
valid_loader = DataLoader(valid_dataset, batch_size=5)

# params
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_reconst = 10.0
lambda_identity = 5.0
gpu_list = list(range(torch.cuda.device_count()))
output_shape = (1, 15, 15)

# define model & init params
generator_AB = GeneratorResNet(num_residual_blocks=1)
generator_BA = GeneratorResNet(num_residual_blocks=1)
discriminator_A = Discriminator()
discriminator_B = Discriminator()

generator_AB.apply(weights_init_normal)
generator_BA.apply(weights_init_normal)
discriminator_A.apply(weights_init_normal)
discriminator_B.apply(weights_init_normal)

# transfer model to cuda
generator_AB, generator_BA, discriminator_A, discriminator_B = apply_multi_gpu_to_model(generator_AB, generator_BA,
                                                                                        discriminator_A, discriminator_B,
                                                                                        gpu_list=gpu_list, is_multi_gpu=True)

# define loss
criterion_valid = nn.MSELoss()  # github: loss_GAN()
criterion_reconst = nn.L1Loss()
criterion_identity = nn.L1Loss()

# define optimizer
g_optimizer = Adam(chain(generator_AB.parameters(), generator_BA.parameters()), lr=lr, betas=(beta1, beta2))
d_A_optimizer = Adam(discriminator_A.parameters(), lr=lr, betas=(beta1, beta2))
d_B_optimizer = Adam(discriminator_B.parameters(), lr=lr, betas=(beta1, beta2))

# define lr_scheduler
g_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(g_optimizer, T_0=50, T_mult=2, eta_min=0.00005)
dA_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(d_A_optimizer, T_0=50, T_mult=2, eta_min=0.00005)
dB_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(d_B_optimizer, T_0=50, T_mult=2, eta_min=0.00005)

# fake buffer
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# train
batch_idx = 1
n_epochs = 200
for epoch in range(n_epochs):
    for real_A, real_B in train_loader:
        # transfer train-data to cuda
        real_A = real_A.cuda()
        real_B = real_B.cuda()

        # make labeling and transfer to cuda
        valid_y = torch.ones(real_A.size(0), *output_shape).cuda()
        fake_y = torch.zeros(real_A.size(0), *output_shape).cuda()

        # ===============
        # Train generator
        # ===============
        # 0. init gradients
        g_optimizer.zero_grad()

        # 1. validation loss: 생성자가 생성한 각 도메인 변환된 이미지를 판별자가 보고 얼마나 정답인지 측정
        fake_A = generator_BA(real_B)
        fake_B = generator_AB(real_A)
        pred_A = discriminator_A(fake_A)
        pred_B = discriminator_B(fake_B)
        valid_loss_BA = criterion_valid(pred_A, valid_y)
        valid_loss_AB = criterion_valid(pred_B, valid_y)
        valid_loss = (valid_loss_BA + valid_loss_AB) / 2

        # 2. reconstrunction loss: 생성자가 생성한 각 도메인 변환된 이미지와 각 도메인의 실제 이미지와 차이가 얼마나 나는지 측정
        reconst_A = generator_BA(fake_B)
        reconst_B = generator_AB(fake_A)
        reconst_loss_A = criterion_identity(reconst_A, real_A)
        reconst_loss_B = criterion_identity(reconst_B, real_B)
        reconst_loss = (reconst_loss_A + reconst_loss_B) / 2

        # 3. identity loss: 타겟 도메인의 실제 이미지를 생성자에 넣었을 때, 타겟 도메인 이미지를 그대로 잘 형성하는지 측정
        identity_A = generator_BA(real_A)
        identity_B = generator_AB(real_B)
        identity_loss_A = criterion_identity(identity_A, real_A)
        identity_loss_B = criterion_identity(identity_B, real_B)
        identity_loss = (identity_loss_A + identity_loss_B) / 2

        # 4. backward & update params
        g_loss = valid_loss + lambda_reconst * reconst_loss + lambda_identity * identity_loss
        g_loss.backward()
        g_optimizer.step()

        # ===================
        # Train discriminator
        # ===================
        # 0. init gradients
        d_A_optimizer.zero_grad()
        d_B_optimizer.zero_grad()

        # 1. discriminaotr A
        real_loss_A = criterion_valid(discriminator_A(real_A), valid_y)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        fake_loss_A = criterion_valid(discriminator_A(fake_A_.detach().clone()), fake_y)
        loss_A = (real_loss_A + fake_loss_A) / 2
        loss_A.backward()
        d_A_optimizer.step()

        # 2. disrcriminator B
        real_loss_B = criterion_valid(discriminator_B(real_B), valid_y)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        fake_loss_B = criterion_valid(discriminator_B(fake_B_.detach().clone()), fake_y)
        loss_B = (real_loss_B + fake_loss_B) / 2
        loss_B.backward()
        d_B_optimizer.step()

        d_loss = (loss_A + loss_B) / 2

        # step lr_scheduler
        g_lr_scheduler.step()
        dA_lr_scheduler.step()
        dB_lr_scheduler.step()

        # save sample image
        if (batch_idx + 1) % 100 == 0:
            sample_images(batch_idx, generator_AB, generator_BA)
        print(f"[Batch {batch_idx}/{len(train_loader)}] [D Loss: {d_loss.item(): .8f}] [G Loss: {g_loss.item(): .8f}(valid-loss: {valid_loss.item(): .8f} | reconst-loss: {reconst_loss.item(): 8f} | identity-loss: {identity_loss.item(): .8f})")
        batch_idx += 1
