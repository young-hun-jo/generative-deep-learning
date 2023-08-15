import os
import pickle
import numpy as np
from typing import List, Tuple

from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Flatten, Dense, Reshape, Activation, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.utils import plot_model
from keras import backend as K

from utils.callbacks import CustomCallback, step_decay_schedule


class AutoEncoder():
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
        self.name = "auto-encoder"
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
        #=========
        # Encoder
        #=========
        encoder_input = Input(shape=self.input_dim, name='encoder-input')
        x = encoder_input
        # Convolution Block
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
        shape_before_flattening: Tuple = K.int_shape(x)[1:]  # shape: (batch, height, width, channel)

        x = Flatten()(x)
        encoder_output = Dense(units=self.z_dim, name='encoder-output')(x)
        self.encoder = Model(inputs=encoder_input, outputs=encoder_output)

        #=========
        # Decoder
        #=========
        decoder_input = Input(shape=(self.z_dim,), name='decoder-input')

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
                x = LeakyReLU()(x)
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:  # LeakyRelu is not applied to final output layer
                x = Activation('sigmoid')(x)

        decoder_output = x
        self.decoder = Model(inputs=decoder_input, outputs=decoder_output)

        # combine encoder and decoder
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(inputs=model_input, outputs=model_output)

    def compile(self, learning_rate: float):
        self.learning_rate = learning_rate
        optimizer = Adam(lr=learning_rate)

        # `RMSE` loss-function
        def r_loss(y, y_):
            return K.mean(K.square(y - y_), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.encoder_conv_filters
                , self.encoder_conv_kernel_size
                , self.encoder_conv_strides
                , self.decoder_conv_t_filters
                , self.decoder_conv_t_kernel_size
                , self.decoder_conv_t_strides
                , self.z_dim
                , self.use_batch_norm
                , self.use_dropout
            ], f)

        self.plot_model(folder)

    def load_weights(self, filepath: str):
        self.model.load_weights(filepath)

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes=True,
                   show_layer_names=True)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint2, custom_callback, lr_sched]

        self.model.fit(
            x_train
            , x_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
        )

