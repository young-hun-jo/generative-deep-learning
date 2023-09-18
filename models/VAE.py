from typing import List

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset


def vae_loss(y_true, y_pred, mu, log_var, r_loss_factor):
    def calc_rmse_loss(y_true, y_pred):
        rmse_loss = torch.mean(torch.square(y_true - y_pred), dim=(1,2,3))
        return rmse_loss

    def calc_kl_loss(mu, log_var):
        kl_loss = -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim=1)
        return kl_loss

    rmse_loss = calc_rmse_loss(y_true, y_pred)
    kl_loss = calc_kl_loss(mu, log_var)
    print("ㄴ rmse loss(mean of batch):", torch.mean(rmse_loss.data))
    print("ㄴ kl loss(mean of batch):", torch.mean(kl_loss.data))
    return torch.mean(r_loss_factor * rmse_loss + kl_loss)


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, mu, log_var):
        epsilon = torch.randn(size=mu.shape)
        return mu + torch.exp(log_var / 2) * epsilon


class VariationalAutoEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 z_dim: int,
                 encoder_hidden_dims: List[int],
                 batch_size):
        super(VariationalAutoEncoder, self).__init__()
        self.batch_size = batch_size
        decoder_hidden_dims = encoder_hidden_dims.copy()
        decoder_hidden_dims.reverse()

        if len(encoder_hidden_dims) != 5:
            raise IndexError("hidden-dims layer length must be five(`5`)")
        # Encoder
        self.encoder_l1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                  out_channels=encoder_hidden_dims[0],
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1),
                                        nn.BatchNorm2d(encoder_hidden_dims[0]),
                                        nn.LeakyReLU()
                                        )
        self.encoder_l2 = nn.Sequential(nn.Conv2d(in_channels=encoder_hidden_dims[0],
                                                  out_channels=encoder_hidden_dims[1],
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1),
                                        nn.BatchNorm2d(encoder_hidden_dims[1]),
                                        nn.LeakyReLU()
                                        )
        self.encoder_l3 = nn.Sequential(nn.Conv2d(in_channels=encoder_hidden_dims[1],
                                                  out_channels=encoder_hidden_dims[2],
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1),
                                        nn.BatchNorm2d(encoder_hidden_dims[2]),
                                        nn.LeakyReLU()
                                        )
        self.encoder_l4 = nn.Sequential(nn.Conv2d(in_channels=encoder_hidden_dims[2],
                                                  out_channels=encoder_hidden_dims[3],
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1),
                                        nn.BatchNorm2d(encoder_hidden_dims[3]),
                                        nn.LeakyReLU()
                                        )
        self.encoder_l5 = nn.Sequential(nn.Conv2d(in_channels=encoder_hidden_dims[3],
                                                  out_channels=encoder_hidden_dims[4],
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1),
                                        nn.BatchNorm2d(encoder_hidden_dims[4]),
                                        nn.LeakyReLU()
                                        )
        # caching variable
        self.shape_before_flatten = (encoder_hidden_dims[-1], 1, 1)
        self.shape_after_flatten = np.prod(self.shape_before_flatten)

        self.mu = nn.Linear(in_features=self.shape_after_flatten, out_features=z_dim)
        self.log_var = nn.Linear(in_features=self.shape_after_flatten, out_features=z_dim)
        self.sampling_layer = Sampling()

        # Decoder
        print("shape_after_flatten:", self.shape_after_flatten)
        self.decoder_fc = nn.Linear(in_features=z_dim, out_features=self.shape_after_flatten)

        self.decoder_l1 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.shape_before_flatten[0],
                                                           out_channels=decoder_hidden_dims[0],
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1),
                                        nn.BatchNorm2d(decoder_hidden_dims[0]),
                                        nn.LeakyReLU()
                                        )
        self.decoder_l2 = nn.Sequential(nn.ConvTranspose2d(in_channels=decoder_hidden_dims[0],
                                                           out_channels=decoder_hidden_dims[1],
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1),
                                        nn.BatchNorm2d(decoder_hidden_dims[1]),
                                        nn.LeakyReLU()
                                        )
        self.decoder_l3 = nn.Sequential(nn.ConvTranspose2d(in_channels=decoder_hidden_dims[1],
                                                           out_channels=decoder_hidden_dims[2],
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1),
                                        nn.BatchNorm2d(decoder_hidden_dims[2]),
                                        nn.LeakyReLU()
                                        )
        self.decoder_l4 = nn.Sequential(nn.ConvTranspose2d(in_channels=decoder_hidden_dims[2],
                                                           out_channels=decoder_hidden_dims[3],
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1),
                                        nn.BatchNorm2d(decoder_hidden_dims[3]),
                                        nn.LeakyReLU()
                                        )
        self.decoder_l5 = nn.Sequential(nn.ConvTranspose2d(in_channels=decoder_hidden_dims[3],
                                                           out_channels=decoder_hidden_dims[4],
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1),
                                        nn.BatchNorm2d(decoder_hidden_dims[4]),
                                        nn.LeakyReLU()
                                        )
        self.final_layer = nn.Sequential(nn.Conv2d(in_channels=decoder_hidden_dims[4],
                                                   out_channels=1,
                                                   kernel_size=3,
                                                   padding='same'),
                                         nn.Sigmoid()
                                         )

    def forward(self, x):
        # Encoder
        x = self.encoder_l1(x)
        x = self.encoder_l2(x)
        x = self.encoder_l3(x)
        x = self.encoder_l4(x)
        x = self.encoder_l5(x)
        x = nn.Flatten()(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        x = self.sampling_layer(mu, log_var)

        # Decoder
        x = self.decoder_fc(x)
        x = torch.reshape(x, (self.batch_size, *self.shape_before_flatten))
        x = self.decoder_l1(x)
        x = self.decoder_l2(x)
        x = self.decoder_l3(x)
        x = self.decoder_l4(x)
        x = self.decoder_l5(x)
        y = self.final_layer(x)

        self.mu_unit = mu
        self.log_var_unit = log_var

        return y


class CustomDataset(Dataset):
    def __init__(self, x: np.array, y: np.array):
        self.x = np.transpose(x, (0, 3, 1, 2))
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index,:,:,:]
        y = self.y[index]
        return x, y