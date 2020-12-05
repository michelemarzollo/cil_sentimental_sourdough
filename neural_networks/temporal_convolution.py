"""
Code inspired from https://github.com/YuanTingHsieh/TF_TCN, a tensorflow implementation
of the paper https://arxiv.org/abs/1803.01271.
Additions to the paper:
    - attention: https://arxiv.org/pdf/1706.03762.pdf
    - gated linear units: https://arxiv.org/pdf/1612.08083.pdf
    - highway networks: https://arxiv.org/pdf/1505.00387.pdf
"""
from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Layer, SpatialDropout1D
from tensorflow.keras import initializers

from neural_networks.transformer import MultiHeadSelfAttention


class WeightNormConv1D(Layer):
    """A dilated one dimensional convolution. Uses weight normalization (Salimans & Kingma, 2016),
    and gives the possibility of using gated linear units (Dauphin et al., 2017) instead of classical ReLUs.
    NOTE: In this implementation input is assumed to be pre-padded"""

    def __init__(self, num_filters, filter_size, dilation_rate, stride=1,
                 pad='VALID', gated=False, keep_len=True, causal=False, **kwargs):
        super(WeightNormConv1D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation_rate = dilation_rate
        self.stride = stride
        self.pad = pad
        self.gated = gated
        self.keep_len = keep_len
        self.causal = causal

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if self.gated:
            self.num_filters = self.num_filters * 2
        # size of V is L, Cin, Cout
        self.V = self.add_weight(shape=(self.filter_size, input_shape[-1], self.num_filters),
                                 dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01),
                                 trainable=True)
        self.g = self.add_weight(shape=(self.num_filters,), dtype=tf.float32,
                                 initializer=initializers.constant(1.), trainable=True)
        self.b = self.add_weight(shape=(self.num_filters,), dtype=tf.float32, initializer=None, trainable=True)

    @staticmethod
    def temporal_padding(x, padding):
        """Pads the middle dimension of a 3D tensor.
        # Arguments
            :param x: Tensor or variable.
            :param padding: Tuple of 2 integers, how many zeros to
                add at the start and end of dim 1.
        # Returns
            :return: A padded 3D tensor.
        """
        assert len(padding) == 2
        # first and third dimension have no padding
        pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
        return tf.pad(tensor=x, paddings=pattern)

    @tf.function
    def call(self, inputs, training=False):
        """
        :param inputs: tensor of shape [N, L, Cin]
        :param training: boolean, true if training mode, false if inference mode (not used)
        :return: tensor of shape [N, L_new, self.num_filters],
                    L_new = L if self.keep_len==True
                    L_new = L - self.dilation_rate*(self.filter_size-1) if self.keep_len==False
        """
        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(self.g, [1, 1, self.num_filters]) * tf.keras.backend.l2_normalize(self.V, [0, 1])
        if self.keep_len:
            # pad x for causal or non causal convolution
            if self.causal:
                left_pad = self.dilation_rate * (self.filter_size - 1)
                right_pad = 0
            else:
                left_pad = self.dilation_rate * tf.math.floor((self.filter_size - 1) / 2)
                right_pad = self.dilation_rate * tf.math.ceil((self.filter_size - 1) / 2)
            x = self.temporal_padding(inputs, (left_pad, right_pad))
        else:
            x = inputs

        # calculate convolutional layer output
        x = tf.nn.bias_add(
            tf.nn.convolution(input=x, filters=W, padding=self.pad, strides=self.stride,
                              dilations=[self.dilation_rate]), self.b)
        # GLU
        if self.gated:
            split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
            split1 = tf.sigmoid(split1)
            x = tf.multiply(split0, split1)
        # ReLU
        else:
            # apply nonlinearity
            x = tf.nn.relu(x)

        return x

    def get_config(self):
        config = super(WeightNormConv1D, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'dilation_rate': self.dilation_rate,
            'stride': self.stride,
            'pad': self.pad,
            'gated': self.gated,
            'keep_len': self.keep_len,
            'causal': self.causal,
        })
        return config

class TemporalBlock(Layer):
    """
    Temporal block in TCN (Bai 2018).
    Possibility to add attention network, and highway connections (Srivastava et al., 2015)
    """

    def __init__(self, out_channels, filter_size, dilation_rate, stride, dropout,
                 use_highway, high_init, gated, attention, attention_dropout, keep_len, causal, **kwargs):
        """
        :param out_channels: number of output channels (number of filters to apply)
        :param filter_size: receptive field of a conv. filter
        :param dilation_rate: holes inbetween
        :param stride: same as what's need in conv. function
        :param dropout: prob. to drop weights
        :param use_highway: (not in original paper) use highway as residual connection
        :param high_init: initialization value for highway connection, if 0 default is used (suggested negative value)
        :param gated: (not in original paper) use gated linear unit as activation
        :param attention: (not in original paper) add self attention block after Conv.
        :param attention_dropout: dropout rate for eventual attention block
        :param causal: whether the 1D convolution should be causal (consider only the past for each word) or not
        """
        super(TemporalBlock, self).__init__(**kwargs)

        self.out_channels = out_channels
        self.use_highway = use_highway
        self.high_init = high_init
        self.attention = attention
        self.keep_len = keep_len

        self.conv_1 = WeightNormConv1D(out_channels, filter_size, dilation_rate,
                                       stride, gated=gated, keep_len=keep_len, causal=causal)
        self.dropout_1 = SpatialDropout1D(dropout)
        self.conv_2 = WeightNormConv1D(out_channels, filter_size, dilation_rate, stride,
                                       gated=gated, keep_len=keep_len, causal=causal)
        self.dropout_2 = SpatialDropout1D(dropout)
        if self.attention:
            self.attention_1 = MultiHeadSelfAttention()
            self.attention_2 = MultiHeadSelfAttention()

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if self.keep_len:
            if self.use_highway:
                self.W_t = self.add_weight(shape=(1, input_shape[-1], self.out_channels), dtype=tf.float32,
                                           initializer=initializers.RandomNormal(0, 0.01), trainable=True)

                self.b_t = self.add_weight(shape=(self.out_channels,), dtype=tf.float32, trainable=True,
                                           initializer=(initializers.Constant(self.high_init) if self.high_init != 0 else None))
                self.b_h = self.add_weight(shape=(self.out_channels,), dtype=tf.float32, trainable=True)

            if input_shape[-1] != self.out_channels:
                self.W_r = self.add_weight(shape=(1, input_shape[-1], self.out_channels), dtype=tf.float32,
                                           initializer=initializers.RandomNormal(0, 0.01), trainable=True)
                self.b_r = self.add_weight(shape=(self.out_channels,), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, inputs, training=False):
        """
        :param inputs: A tensor of shape [N, L, Cin]
        :param training: boolean, if it's in training mode
        :return: A tensor of shape [N, L, self.out_channels]
        """
        out1 = self.conv_1(inputs)
        out1 = self.dropout_1(out1, training=training)
        if self.attention:
            out1 = self.attention_1(out1)
        out2 = self.conv_2(out1)
        out2 = self.dropout_2(out2, training=training)
        if self.attention:
            out2 = self.attention_2(out2)

        if self.keep_len:
            if self.use_highway:
                H = tf.nn.bias_add(out2, self.b_h)
                T = tf.nn.bias_add(tf.nn.convolution(input=inputs, filters=self.W_t, padding='SAME'), self.b_t)
                T = tf.nn.sigmoid(T)
                # to make shapes match if input and output channels are different
                if inputs.shape[-1] != self.out_channels:
                    inputs = tf.nn.bias_add(tf.nn.convolution(input=inputs, filters=self.W_r, padding='SAME'), self.b_r)
                out2 = H * T + inputs * (1.0 - T)
            elif inputs.shape[-1] != self.out_channels:
                residual = tf.nn.bias_add(tf.nn.convolution(input=inputs, filters=self.W_r, padding='SAME'), self.b_r)
                out2 += residual
            else:
                out2 += inputs

        return tf.nn.relu(out2)

    def get_config(self):
        config = super(TemporalBlock, self).get_config()
        config.update({
            'out_channels': self.out_channels,
            'use_highway': self.use_highway,
            'high_init': self.high_init,
            'attention': self.attention,
            'keep_len': self.keep_len,
        })
        return config


class TCN(Layer):
    """
    A network that stacks several temporal convolution blocks, with exponentially increasing dilation
    rate to increase receptive field.
    """
    """Type hints"""
    temporal_layers: List[TemporalBlock]

    def __init__(self,
                 num_channels,
                 kernel_size,
                 stride,
                 dropout=0.1,
                 use_highway=False,
                 high_init=0,
                 use_glu=False,
                 attention=False,
                 attention_dropout=0.5,
                 keep_len=True,
                 causal=False,
                 propagate_masking=False,
                 **kwargs):
        """
        :param num_channels: Number of channels per layer (list of ints, len is the number of levels)
        :param kernel_size: Kernel size for each level
        :param stride: stride value for each layer (should always be one, since dilations are used)
        :param dropout: dropout rate for spatial dropout in TemporalBlock
        :param use_highway: whether to use highway connections in TemporalBlock
        :param high_init: initialization value for highway connection, if 0 default is used (suggested negative value)
        :param use_glu: whether to use gated linear units instead of ReLUs in TemporalBlock
        :param attention: whether to use attention
        :param attention_dropout: dropout rate for (eventual) attention network
        :param keep_len: if true, all layers will have the same length, by using padding
        :param causal: whether the 1D convolution should be causal (consider only the past for each word) or not
        """
        super(TCN, self).__init__(**kwargs)
        if propagate_masking:
            self.supports_masking = True
        self.num_levels = len(num_channels)
        # creates a list of temporal layers
        self.temporal_layers = []
        for i in range(self.num_levels):
            dilation_size = 2 ** i  # exponential increase of dilation
            self.temporal_layers.append(TemporalBlock(num_channels[i], kernel_size[i], dilation_size,
                                                      stride[i], dropout, use_highway, high_init, use_glu, attention,
                                                      attention_dropout, keep_len, causal))

    @tf.function
    def call(self, inputs, training=None):
        x = inputs
        for i in range(self.num_levels):
            x = self.temporal_layers[i](x)

        # here x has shape [N, L, num_channels[-1]]
        return x

    def get_config(self):
        config = super(TCN, self).get_config()
        config.update({
            'num_levels': self.num_levels,
            'supports_masking': self.supports_masking,
        })
        return config
