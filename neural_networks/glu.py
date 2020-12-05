"""Implementation of the Gated Linear Units from the paper Language Modeling with Gated Convolutional Networks, Y. Dauphin, 2017"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

class GLU(Layer):
    def __init__(self, axis=2):
        super(GLU, self).__init__()
        self.axis = axis
        
    @tf.function
    def call(self, inputs, training=False):
        split_fw0, split_fw1, split_bw0, split_bw1 = tf.split(inputs, num_or_size_splits=4, axis=self.axis)

        split_fw1 = tf.sigmoid(split_fw1)
        split_bw1 = tf.sigmoid(split_bw1)
        
        split_fw = tf.multiply(split_fw0, split_fw1)
        split_bw = tf.multiply(split_bw0, split_bw1)
        return tf.concat( [split_fw, split_bw], axis=self.axis )
        # split0, split1 = tf.split(inputs, num_or_size_splits=2, axis=2)
        # split1 = tf.sigmoid(split1)
        # return tf.multiply(split0, split1)
