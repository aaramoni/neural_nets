import numpy as np
from tensorflow.keras import Model, backend as K
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, \
    Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

class Autoencoder:
    '''Mirrored autoencoder with convolutional layers.'''

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):

        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def compile(self,learning_rate=.0001,optimizer=None,loss=None):
        optimizer = Adam(learning_rate=learning_rate) if optimizer is None else optimizer
        loss = MeanSquaredError() if loss is None else loss
        self.model.compile(optimizer=optimizer,loss=loss)

    def train(self,x_train,batch_size,epochs):
        return self.model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')

    # encoder
    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name='encoder_input')
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def _add_conv_layers(self,x):
        ''' Creates convolutional blocks consisting of: Conv2D + ReLU + Batch Normalization'''
        for layer_index in range(self._num_conv_layers):
            conv_layer = Conv2D(filters=self.conv_filters[layer_index],
                                kernel_size=self.conv_kernels[layer_index],
                                strides=self.conv_strides[layer_index],
                                padding='same',
                                name=f'encoder_conv_layer_{layer_index + 1}')
            x = conv_layer(x)
            x = ReLU(name=f'encoder_relu_{layer_index + 1}')(x)
            x = BatchNormalization(name=f'encoder_bn_{layer_index + 1}')(x)
        return x

    def _add_bottleneck(self,x):
        ''' Creates latent space flattening data and adding Dense layer.'''
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name='encoder_output')(x)
        return x

    # decoder
    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name='decoder_input')
        dense_layer = Dense(np.prod(self._shape_before_bottleneck), name='decoder_dense')(decoder_input)
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_conv_transpose_layers(self,x):
        ''' Creates convolutional transpose blocks consisting on: Conv2DTranspose + ReLU + Batch Normalizationx'''
        for layer_index in reversed(range(1,self._num_conv_layers)):
            conv_transpose_layer = Conv2DTranspose(
                filters=self.conv_filters[layer_index],
                kernel_size=self.conv_kernels[layer_index],
                strides=self.conv_strides[layer_index],
                padding='same',
                name=f'decoder_conv_transpose_layer_{self._num_conv_layers-layer_index}')
            x =  conv_transpose_layer(x)
            x = ReLU(name=f'decoder_relu_{self._num_conv_layers-layer_index}')(x)
            x = BatchNormalization(name=f'decoder_bn_{self._num_conv_layers-layer_index}')(x)
        return x

    def _add_decoder_output(self,x):
        conv_transpose_layer = Conv2DTranspose(
            filters=self.input_shape[2],
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding='same',
            name=f'decoder_conv_transpose_layer_{self._num_conv_layers}')
        x = conv_transpose_layer(x)
        return Activation('sigmoid', name='decoder_output_layer')(x)

    # processing
    def reconstruct(self, data):
        latent_space_repr = self.encoder.predict(data)
        return self.decoder.predict(latent_space_repr), latent_space_repr
