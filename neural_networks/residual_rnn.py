import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, GRU, Bidirectional
from tensorflow.keras import initializers


class ResidualRNN(Layer):
    """A bidirectional RNN layer with residual connections between input and output. The RNN layer can be
    either an LSTM or a GRU. Residuals can be standard or highway connections."""

    def __init__(self,
                 rnn_type='lstm',
                 units=128,
                 dropout=.3,
                 residual='residual',
                 high_init=-1,
                 return_sequences=False,
                 **kwargs):
        """
        :param units: number of units of LSTM layer
        :param dropout: internal dropout of the LSTM layer
        :param residual: string, either 'highway', 'residual' or 'none', to choose whether to use residual connections
            and which kind o use
        :param high_init: used only if residual=='highway'. If 0, glorot uniform initialization, otherwise constant of
            the specified number (suggested negative number, such as -1 or -3)
        :param: rnn_type: string, either 'gru' or 'lstm'
        :param return_sequences: the same as in keras LSTM layer
        :param kwargs: keyword arguments
        """
        super(ResidualRNN, self).__init__(**kwargs)

        self.rnn_type = rnn_type
        self.units = units
        self.residual = residual
        self.high_init = high_init
        self.return_sequences = return_sequences

        if rnn_type == 'gru':
            self.rnn_layer = Bidirectional(GRU(units, dropout=dropout, return_sequences=self.return_sequences))  # True
        else:
            self.rnn_layer = Bidirectional(LSTM(units, dropout=dropout, return_sequences=self.return_sequences))  # True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if self.return_sequences:
            if self.residual == 'highway':
                self.b_h = self.add_weight(shape=(self.units*2,), dtype=tf.float32, trainable=True)
                self.W_t = self.add_weight(shape=(1, input_shape[-1], self.units*2), dtype=tf.float32,
                                           initializer=initializers.RandomNormal(0, 0.01), trainable=True)

                self.b_t = self.add_weight(shape=(self.units*2,), dtype=tf.float32, trainable=True,
                                           initializer=(
                                               initializers.Constant(self.high_init) if self.high_init != 0 else None))

            if self.residual != 'none' and input_shape[-1] != self.units*2:
                self.W_r = self.add_weight(shape=(1, input_shape[-1], self.units*2), dtype=tf.float32,
                                           initializer=initializers.RandomNormal(0, 0.01), trainable=True)

                self.b_r = self.add_weight(shape=(self.units*2,), dtype=tf.float32, trainable=True)

    def compute_mask(self, inputs, mask=None):
        """
        If the layer is not the last one (return_sequences==True), the mask is propagated to the following layer
        """
        if self.return_sequences:
            return mask
        else:
            return None

    @tf.function
    def call(self, inputs, mask=None, training=None):
        out = self.rnn_layer(inputs, mask=mask, training=training)
        if self.return_sequences:
            if self.residual == 'highway':
                H = tf.nn.bias_add(out, self.b_h)
                T = tf.nn.bias_add(tf.nn.convolution(input=inputs, filters=self.W_t, padding='SAME'), self.b_t)
                T = tf.nn.sigmoid(T)
                if inputs.shape[-1] != self.units*2:
                    inputs = tf.nn.bias_add(tf.nn.convolution(input=inputs, filters=self.W_r, padding='SAME'), self.b_r)
                out = H * T + inputs * (1.0 - T)
            elif self.residual == 'residual':
                if inputs.shape[-1] != self.units*2:
                    inputs = tf.nn.bias_add(tf.nn.convolution(input=inputs, filters=self.W_r, padding='SAME'), self.b_r)
                out = out + inputs
            if self.residual != 'none':
                out = tf.nn.relu(out)
        return out
        # Alternative if you want to use residual connections also on the last layer
        # if self.return_sequences:
        #     return out
        # else:
        #     # (i think this only learns one of the two directions), in case we want to have higway also with
        #     # return_sequences=False
        #     return tf.reshape(out[:, 0, :], (-1, self.units*2))  # -1 for pre-padding, 0 for post-padding


    def get_config(self):
        config = super(ResidualRNN, self).get_config()
        config.update({
            'rnn_type': self.rnn_type,
            'units': self.units,
            'residual': self.residual,
            'high_init': self.high_init,
            'return_sequences': self.return_sequences,
        })
        return config
