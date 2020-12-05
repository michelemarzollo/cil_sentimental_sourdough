import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras import initializers

"""
Implementation of the Attention Layer from the paper "Hierarchical Attention Networks for Document Classiﬁcation", Zichao Yang, 2016
    with additions from the scaled-dot-product as in "Attention is all you need", A. Vaswani, 2017
"""

class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.internal_size = input_shape[-1]

        self.W_h = self.add_weight(shape=(1, self.internal_size, self.internal_size), dtype=tf.float32,
                                   initializer=initializers.RandomNormal(0, 0.01), trainable=True)

        self.b_h = self.add_weight(shape=(self.internal_size,), dtype=tf.float32, trainable=True)

        self.u_w = self.add_weight(shape=(self.internal_size,), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, inputs, training=False):
        # u : hidden word representation (1 layer MLP)
        u = tf.tanh(tf.nn.bias_add(tf.linalg.matmul(inputs, self.W_h), self.b_h))
        # weight alpha : word-similarity to word-reference u_w (trainable)
        alpha = tf.nn.softmax(tf.linalg.matvec(u, self.u_w) / tf.sqrt(tf.cast(self.internal_size, tf.float32)))

        return tf.linalg.matvec(inputs, alpha,
                                transpose_a=True)  # tf.reduce_sum( tf.expand_dims(alpha, 2) * inputs, axis=1)
"""
Implementation of a MultiHead Attention inspired from the MultiHead Self-Attention in "Transformer as in Attention is all you need", A. Vaswani, 2017
"""
class MultiHead_Attention(Layer):
    def __init__(self,
                    n_head=8,
                    inner_dims_expansion=2,
                    use_glu_head=False,
                    use_glu_out=False,
                    internal_dropout=.3,
                    exit_dropout=.1):
        """
        :param n_head: number of heads in the multi-head self-attention
        :param inner_dims_expansion: each head will have a dimension of embedding_size*inner_dims_expansion/n_head
        :param use_glu_head: use glu for each head
        :param use_glu_out: use glu at the end of the self-attention layer
        :param internal_dropout: dropout within the attention layer
        :param exit_dropout: dropout at the end of the attention layer
        """
        super(MultiHead_Attention, self).__init__()
        self.n_head = n_head
        self.use_glu_head = use_glu_head
        self.use_glu_out = use_glu_out
        self.dropout_attn_q = Dropout(internal_dropout)
        self.dropout_attn_k = Dropout(internal_dropout)
        self.dropout_attn_v = Dropout(internal_dropout)
        self.dropout_attn_out = Dropout(internal_dropout)
        self.dropout_multihead_layer = Dropout(exit_dropout)
        self.inner_dims_expansion = inner_dims_expansion

    def build(self, input_shape):
        self.d_model = input_shape[-1]  # embedding_size
        self.d_k = self.d_v = self.inner_dims_expansion * self.d_model // self.n_head
        if self.use_glu_head:
            self.d_v *= 2

        self.d_o = self.d_model
        if self.use_glu_out:
            self.d_o *= 2

        self.u_w = self.add_weight(shape=(self.d_k * self.n_head,), dtype=tf.float32, trainable=True)

        self.W_k = self.add_weight(shape=(1, self.d_model, self.d_k * self.n_head), dtype=tf.float32,
                                   initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        self.W_v = self.add_weight(shape=(1, self.d_model, self.d_v * self.n_head), dtype=tf.float32,
                                   initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        self.W_o = self.add_weight(shape=(1, self.d_v * self.n_head, self.d_o), dtype=tf.float32,
                                   initializer=initializers.RandomNormal(0, 0.01), trainable=True)

        self.b_k = self.add_weight(shape=(self.d_k * self.n_head,), dtype=tf.float32,
                                   initializer=initializers.Constant(0), trainable=True)
        self.b_v = self.add_weight(shape=(self.d_v * self.n_head,), dtype=tf.float32,
                                   initializer=initializers.Constant(0), trainable=True)
        self.b_o = self.add_weight(shape=(self.d_o,), dtype=tf.float32, initializer=initializers.Constant(0),
                                   trainable=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None, training=False):
        """
        Args:
            Q (tf.tensor): of shape (n_head*batch, d_k)
            K/V (tf.tensor): of shape (n_head*batch, seq_len, d_k/d_v)
            mask  (tf.tensor): of shape (n_head*batch, seq_len)
        Returns:
            a tensor of shape (n_head*batch, d_v)
        """

        # [num_head*batch, seq_len]
        alpha = tf.linalg.matvec(K, Q)
        alpha = alpha / tf.sqrt(tf.cast(self.d_k, tf.float32))

        if mask is not None:
            # [num_head*batch, seq_len]
            mask = tf.cast(mask, tf.float32)
            alpha = tf.multiply(alpha, mask) + (1.0 - mask) * (-1e10)

        alpha = tf.nn.softmax(alpha)

        # XXX
        #        if training:
        #            alpha = self.dropout_attn_layer(alpha)

        # [num_head*batch, d_v]
        return tf.linalg.matvec(V, alpha, transpose_a=True)  # tf.reduce_sum( tf.expand_dims(alpha, 2) * V, axis=1)

    @staticmethod
    def glu(inputs):
        split_0, split_1 = tf.split(inputs, num_or_size_splits=2, axis=1)

        split_1 = tf.sigmoid(split_1)
        return tf.multiply(split_0, split_1)

    @tf.function
    def call(self, inputs, mask=None, training=False):
        """
        Args:
            inputs (tf.tensor): of shape (batch, seq_len, d_model)
            mask   (tf.tensor): of shape (batch, seq_len)
        Returns:
            a tensor of shape (batch, seq_len, d_model)
        """

        # [batch_size, seq_len, n_head*d_k]
        Q = tf.tile(tf.expand_dims(self.u_w, 0), [tf.shape(inputs)[0], 1])
        K = tf.nn.bias_add(tf.matmul(inputs, self.W_k), self.b_k)  # no activation, no bias ?? XXX
        V = tf.nn.bias_add(tf.matmul(inputs, self.W_v), self.b_v)
        if training:
            K = self.dropout_attn_k(K)
            V = self.dropout_attn_v(V)

        # [n_head*batch_size, seq_len, d_k/d_v]
        Q_split = tf.concat(tf.split(Q, num_or_size_splits=self.n_head, axis=1), axis=0)
        K_split = tf.concat(tf.split(K, num_or_size_splits=self.n_head, axis=2), axis=0)
        V_split = tf.concat(tf.split(V, num_or_size_splits=self.n_head, axis=2), axis=0)

        if mask is not None:
            mask = tf.tile(mask, [self.n_head, 1])

        # Apply scaled dot product attention
        out = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask)
        print('out', out.shape)
        if self.use_glu_head:
            out = self.glu(out)

        if training:
            out = self.dropout_attn_out(out)

        # Merge the multi-head back to the original shape
        # [batch, d_v*n_head]
        out = tf.concat(tf.split(out, self.n_head, axis=0), axis=1)

        # [batch, seq_len, d_model]
        out = tf.nn.bias_add(tf.linalg.matvec(self.W_o, out, transpose_a=True), self.b_o)

        if self.use_glu_out:
            out = self.glu(out)

        if training:
            out = self.dropout_multihead_layer(out)
        return out
