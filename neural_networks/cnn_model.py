"""Implementation similar to the paper Convolutional Neural Networks for Sentence Classification, Kim Y., 2014
In addition dilations are added."""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D


class CNN(Layer):
    """Processes the input sequence of size [batch_size, seq_len, emb_size] with 1D convolutions with
    filters of different sizes and dilations. Then applies max-pooling."""
    def __init__(self, filters, kernel_sizes, strides, dilation_rates, **kwargs):
        """
        :param filters: list of integers, number of filters for each filter type
        :param kernel_sizes: list of integers, size of each kernel/filter
        :param strides: list of integers, strides for each filter, suggested always [1]*len(filters)
        :param dilation_rates: list of integers, dilation rated for each kind of filter
        :param kwargs: keyword arguments
        """
        super(CNN, self).__init__(**kwargs)
        self.num_convolutions = len(filters)
        assert len(kernel_sizes) == self.num_convolutions and len(strides) == self.num_convolutions and \
            len(dilation_rates) == self.num_convolutions, "All input tuples must have the same length"

        self.convolutions = []
        for i in range(self.num_convolutions):
            self.convolutions.append(Conv1D(filters[i], kernel_sizes[i], strides[i], padding='same',
                                            dilation_rate=dilation_rates[i], activation='relu', use_bias=True))

    @tf.function
    def call(self, inputs, training=None):
        max_pooled = None
        for i in range(self.num_convolutions):
            x = self.convolutions[i](inputs)
            # x's shape is [batch_size, steps(num_words), filters]
            x = tf.reduce_max(x, axis=1, keepdims=False)
            # now x's shape is [batch_size, filters]
            
            if max_pooled is None:
                max_pooled = x
            else:
                max_pooled = tf.concat([max_pooled, x], axis=1)
        return max_pooled

    def get_config(self):
        config = super(CNN, self).get_config()
        config.update({
            'num_convolutions': self.num_convolutions
        })
        return config
