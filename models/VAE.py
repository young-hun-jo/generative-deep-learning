from typing import Tuple, List

import numpy as np

from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dropout, Flatten, Dense, Lambda, Reshape, Activation
from keras.layers import Conv2DTranspose
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K


class VariationalAutoEncoder(object):
    def __init__(self,
                 input_dim: Tuple,
                 encoder_conv_filters: List[int],
                 encoder_conv_kernel_size: List[int],
                 encoder_conv_strides: List[int],
                 decoder_conv_t_filters: List[int],
                 decoder_conv_t_kernel_size: List[int],
                 decoder_conv_t_strides: List[int],
                 z_dim: int,
                 use_batch_norm=False,
                 use_dropout=False
                 ):
        self.name = "variational-auto-encoder"
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.z_dim = z_dim

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self._build()

    def _build(self):
        # =======
        # Encoder
        # =======
        encoder_input = Input(shape=self.input_dim, name='encoder-input')
        x = encoder_input
        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(filters=self.encoder_conv_filters[i],
                                kernel_size=self.encoder_conv_kernel_size[i],
                                strides=self.encoder_conv_strides[i],
                                padding='same',
                                name=f'encoder_conv_{i}')
            x = conv_layer(x)
            x = LeakyReLU()(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)
        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        # make layers for sampling layer
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)
        self.encoder_mu_log_var = Model(inputs=encoder_input, outputs=(self.mu, self.log_var))

        # make sampling layer using Lambda-layer
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0, stddev=1.0)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(inputs=encoder_input, outputs=encoder_output)

        # =======
        # Decoder
        # =======
        decoder_input = Input(shape=(self.z_dim, 1), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)
        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                                           kernel_size=self.decoder_conv_t_kernel_size[i],
                                           strides=self.decoder_conv_t_strides[i],
                                           padding='same',
                                           name=f'decoder_conv_t_{i}')

            x = conv_t_layer(x)
            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:  # LeakyRelu is not applied to final output layer
                x = Activation('sigmoid')(x)

        decoder_output = x
        self.decoder = Model(inputs=decoder_input, outputs=decoder_output)

        # make full vae
        model_input = encoder_input
        model_output = self.decoder(model_input)
        self.model = Model(inputs=model_input, outputs=model_output)

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        # reconstruction loss
        def vae_rmse_loss(y_true, y_pred):
            rmse_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * rmse_loss
        # latent loss
        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
            return kl_loss
        # final loss of vae
        def vae_loss(y_true, y_pred):
            rmse_loss = vae_rmse_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return rmse_loss + kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_rmse_loss, vae_kl_loss])





