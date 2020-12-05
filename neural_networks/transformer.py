"""
Implementation of the encoder of the "Transformer as in Attention is all you need", A. Vaswani, 2017,
    with additions from :
        - "Character-level language modeling with deeper self-attention",  R. Al-Rfou, 2019
        - "Bert:Pre-training of deep bidirectional transformers for language understanding", J. Devlin, 2018
    [ implementation inspired from https://raw.githubusercontent.com/Lsdefine/attention-is-all-you-need-keras ]
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, LayerNormalization, Dropout
from tensorflow.keras import initializers

def glu(inputs):
    split_0, split_1 = tf.split(inputs, num_or_size_splits=2, axis=2)

    split_1 = tf.sigmoid(split_1)
    return tf.multiply(split_0, split_1)

class MultiHeadSelfAttention(Layer):
    """
    A multi-head self-attention layer,
        - inputs to each head are obtained by equally splitting the embeddings
        - for each head queries, keys, values are obtained from a linear projection of the inputs
        - each head then performs the self-attention function in parallel, yielding an output
        - outputs of each head are concatenated back together, and projected to the final results
    """
    def __init__(self, n_head=8, inner_dims_expansion=2, use_glu_head=False, use_glu_out=False, internal_dropout=.3, exit_dropout=.1):
        super(MultiHeadSelfAttention, self).__init__()
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
        self.d_model = input_shape[-1] #embedding_size
#        assert( self.inner_dims_expansion * self.d_model % self.n_head == 0 )
        
        self.d_k = self.d_v1 = self.d_v2 = self.inner_dims_expansion * self.d_model // self.n_head
        if self.use_glu_head:
            self.d_v1 *= 2

        self.d_o = self.d_model
        if self.use_glu_out:
            self.d_o *= 2

        self.W_q = self.add_weight(shape=(1, self.d_model, self.d_k * self.n_head), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        self.W_k = self.add_weight(shape=(1, self.d_model, self.d_k * self.n_head), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        self.W_v = self.add_weight(shape=(1, self.d_model, self.d_v1 * self.n_head), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        self.W_o = self.add_weight(shape=(1, self.d_v2 * self.n_head, self.d_o), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        
        self.b_q = self.add_weight(shape=(self.d_k * self.n_head,), dtype=tf.float32, initializer=initializers.Constant(0), trainable=True)
        self.b_k = self.add_weight(shape=(self.d_k * self.n_head,), dtype=tf.float32, initializer=initializers.Constant(0), trainable=True)
        self.b_v = self.add_weight(shape=(self.d_v1 * self.n_head,), dtype=tf.float32, initializer=initializers.Constant(0), trainable=True)
        self.b_o = self.add_weight(shape=(self.d_o,), dtype=tf.float32, initializer=initializers.Constant(0), trainable=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None, training=False):
        """
        Args:
            Q/K/V (tf.tensor): of shape (n_head*batch, seq_len, d_k/d_k/d_v)
            mask  (tf.tensor): of shape (n_head*batch, seq_len)
        Returns:
            a tensor of shape (n_head*batch, seq_len, d_v)
        """
        
         # [num_head*batch, seq_len, seq_len]
        alpha = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        alpha = alpha / tf.sqrt(tf.cast(self.d_k, tf.float32))
        
        if mask is not None:
             # [num_head*batch, seq_len, seq_len]
            mask = tf.tile(tf.expand_dims(mask, 1), [1, mask.shape[-1], 1])
            mask = tf.cast(mask, tf.float32)
            alpha = tf.multiply(alpha, mask) + (1.0 - mask) * (-1e10)
        
        alpha = tf.nn.softmax(alpha)
        
        out = tf.matmul(alpha, V) # [num_head*batch, seq_len, d_v]
        return out

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
        Q = tf.nn.bias_add(tf.matmul(inputs, self.W_q), self.b_q)
        K = tf.nn.bias_add(tf.matmul(inputs, self.W_k), self.b_k)
        V = tf.nn.bias_add(tf.matmul(inputs, self.W_v), self.b_v)
        if training:
            Q = self.dropout_attn_q(Q)
            K = self.dropout_attn_k(K)
            V = self.dropout_attn_v(V)
    
         # [n_head*batch_size, seq_len, d_k/d_v]
        Q_split = tf.concat(tf.split(Q, num_or_size_splits=self.n_head, axis=2), axis=0)
        K_split = tf.concat(tf.split(K, num_or_size_splits=self.n_head, axis=2), axis=0)
        V_split = tf.concat(tf.split(V, num_or_size_splits=self.n_head, axis=2), axis=0)

        if mask is not None:
            mask = tf.tile(mask, [self.n_head, 1])
        
        # Apply scaled dot product attention
        out = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask)
        if self.use_glu_head:
            out = glu(out)
        
        if training:
            out = self.dropout_attn_out(out)

        # Merge the multi-head back to the original shape
         # [batch, seq_len, d_v*n_head]
        out = tf.concat(tf.split(out, self.n_head, axis=0), axis=2)

         # [batch, seq_len, d_model]
        out = tf.nn.bias_add(tf.matmul(out, self.W_o), self.b_o)
        
        if self.use_glu_out:
            out = glu(out)
        
        if training:
            out = self.dropout_multihead_layer(out)
        return out

class Positionwise_FeedForward(Layer):
    """
    A feedforward layer composed of two linear transformations with a ReLU activation in between
    """
    def __init__(self, inner_dims_expansion, use_glu=True, dropout=0.3):
        super(Positionwise_FeedForward, self).__init__()
        self.inner_dims_expansion = inner_dims_expansion
        self.use_glu = use_glu
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)

    def build(self, input_shape):
        self.d_model = input_shape[-1] #embedding_size
        self.inner_dims = self.d_model*self.inner_dims_expansion
        
        self.W_1 = self.add_weight(shape=(1, self.d_model, self.inner_dims), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        self.b_1 = self.add_weight(shape=(self.inner_dims,), dtype=tf.float32, trainable=True)
        
        layer2_output_dim = self.d_model
        if self.use_glu:
            layer2_output_dim *= 2
            
        self.W_2 = self.add_weight(shape=(1, self.inner_dims, layer2_output_dim), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        self.b_2 = self.add_weight(shape=(layer2_output_dim,), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, inputs, training=False):
        """
        Args:
            inputs (tf.tensor): of shape (batch, seq_len, d_model)
        Returns:
            a tensor of shape (batch, seq_len, d_model)
        """
        out = tf.nn.bias_add(tf.matmul(inputs, self.W_1), self.b_1)
        out = tf.nn.relu(out)
        if training:
            out = self.dropout_1(out)

        out = tf.nn.bias_add(tf.matmul(out, self.W_2), self.b_2)
        if training:
            out = self.dropout_2(out)

        if self.use_glu:
            out = glu(out)
        
        return out

class Hidden_Layer(Layer):
    """
    A single hidden layers composed of:
        - multi-head self-attention layer
        - feedforward layer
    """
    def __init__(self,
                n_head,
                attn_inner_dim_expansion=2,
                feedfw_inner_dim_expansion=8,
                attn_use_glu_head=False,
                attn_use_glu_out=False,
                feedfw_use_glu=False,
                dropout_in=.1,
                dropout_attn_intern=.1,
                dropout_attn_exit=.3,
                dropout_feedfw=.3,
                dropout_out=.5,
                depth=0,
                initlz_pass_branch=False,
                add_pos=True):
        super(Hidden_Layer, self).__init__()
        self.depth = depth
        self.initlz_pass_branch = initlz_pass_branch
        self.add_pos = add_pos

        self.multi_head_attn_layer = MultiHeadSelfAttention(n_head=n_head,
                                                            inner_dims_expansion=attn_inner_dim_expansion,
                                                            use_glu_head=attn_use_glu_head,
                                                            use_glu_out=attn_use_glu_out,
                                                            internal_dropout=dropout_attn_intern,
                                                            exit_dropout=dropout_attn_exit)

        self.feedfw_layer = Positionwise_FeedForward(inner_dims_expansion=feedfw_inner_dim_expansion,
                                                        use_glu=feedfw_use_glu,
                                                        dropout=dropout_feedfw)
        
        self.norm_layer_attn = LayerNormalization()
        self.norm_layer_feedfw = LayerNormalization()

        self.dropout_in = Dropout(dropout_in)
        self.dropout_attn = Dropout(dropout_out)
        self.dropout_feedfw = Dropout(dropout_out)

    def build(self, input_shape):
        self.d_model = input_shape[-1] #embedding_size
        self.seq_len = input_shape[-2]
        
        self.W_t_attn = self.add_weight(shape=(1, self.d_model, self.d_model), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        if self.initlz_pass_branch:
            self.b_t_attn = self.add_weight(shape=(self.d_model,), dtype=tf.float32, initializer=initializers.Constant(3), trainable=True)
        else:
            self.b_t_attn = self.add_weight(shape=(self.d_model,), dtype=tf.float32, initializer=initializers.Constant(-3), trainable=True)
            
        
        self.W_t_feedfw = self.add_weight(shape=(1, self.d_model, self.d_model), dtype=tf.float32, initializer=initializers.RandomNormal(0, 0.01), trainable=True)
        if self.initlz_pass_branch:
            self.b_t_feedfw = self.add_weight(shape=(self.d_model,), dtype=tf.float32, initializer=initializers.Constant(-3), trainable=True)
        else:
            self.b_t_feedfw = self.add_weight(shape=(self.d_model,), dtype=tf.float32, initializer=initializers.Constant(-3), trainable=True)

        if self.add_pos:
            if self.depth == 0:
                # [seq_len, d_model]
                self.pos_enc_matrix = np.array([ [pos / np.power(10000, 2 * (j // 2) / self.d_model) for j in range(self.d_model)] for pos in range(self.seq_len) ])
                self.pos_enc_matrix[:, 0::2] = np.sin(self.pos_enc_matrix[:, 0::2]) # dim 2i
                self.pos_enc_matrix[:, 1::2] = np.cos(self.pos_enc_matrix[:, 1::2]) # dim 2i+1
                # [1, seq_len, d_model]
                self.pos_enc_matrix = [ self.pos_enc_matrix ]
            else:
                self.pos_enc_matrix = np.array([ np.zeros(self.d_model) for pos in range(self.seq_len) ])
            
            self.pos_enc_weight = self.add_weight(shape=(1, self.seq_len, self.d_model), dtype=tf.float32, initializer=initializers.Constant(self.pos_enc_matrix), trainable=True)

    def add_pos_encoding(self, inputs):
        return tf.add(inputs, self.pos_enc_weight)

    @tf.function
    def call(self, inputs, mask=None, training=False):
        if self.add_pos:
            out = self.add_pos_encoding(inputs)
        else:
            out = inputs
        if training:
            out = self.dropout_in(out)
        
        # One multi-head attention
        T_attn = tf.nn.bias_add(tf.matmul(out, self.W_t_attn), self.b_t_attn)
        T_attn = tf.nn.sigmoid(T_attn)
        
        out = self.multi_head_attn_layer(out, mask=mask) * T_attn + out * (1.0 - T_attn)
        out = self.norm_layer_attn(out)
        if training:
            out = self.dropout_attn(out)
        
        # + One feed-forword
        T_feedfw = tf.nn.bias_add(tf.matmul(out, self.W_t_feedfw), self.b_t_feedfw)
        T_feedfw = tf.nn.sigmoid(T_feedfw)
        
        out = self.feedfw_layer(out) * T_feedfw + out * (1.0 - T_feedfw)
        out = self.norm_layer_feedfw(out)
        if training:
            out = self.dropout_feedfw(out)
        
        return out

class Transformer(Layer):
    """
    A network that stacks several hidden layers composed of:
        - multi-head self-attention layer
        - feedforward layer
    """
    def __init__(self,
                num_layers=6,
                n_head=8,
                attn_inner_dim_expansion=2,
                feedfw_inner_dim_expansion=8,
                attn_use_glu_head=False,
                attn_use_glu_out=False,
                feedfw_use_glu=False,
                dropout_hl_in=.1,
                dropout_attn_intern=.1,
                dropout_attn_exit=.3,
                dropout_feedfw=.3,
                dropout_hl_out=.5,
                return_sequences=False,
                add_pos_encoding=True):
        """
        :param num_layers: number of hidden layers
        :param n_head: number of heads in the multi-head self-attention
        :param attn_inner_dim_expansion: (not in original paper) each head will have a dimension of embedding_size*attn_inner_dim_expansion/n_head
        :param feedfw_inner_dim_expansion: the inner layer in the feedforward layer will have a dimension of embedding_size*feedfw_inner_dim_expansion
        :param attn_use_glu_head: (not in original paper) use glu for each head
        :param attn_use_glu_out: (not in original paper) use glu at the end of the self-attention layer
        :param feedfw_use_glu: (not in original paper) use glu at the end of the feedforward layer
        :param dropout_hl_in: dropout at the beginning of the hidden layer
        :param dropout_attn_intern: dropout within the self-attention layer
        :param dropout_attn_exit: dropout at the end of the self-attention layer
        :param dropout_feedfw: dropout in the feedforward layer
        :param dropout_hl_out: dropout at the end of the hidden layer
        :param return_sequences: (not in original paper) whether we should return a sequence of the same len as the input one,
                                    or a single value describing the whole sequence
        :param add_pos_encoding: (not in original paper) whether we should add a positional encoding at each hidden layer, or only at the first one
        """

        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.layers = [ Hidden_Layer(n_head=n_head,
                                        attn_inner_dim_expansion=attn_inner_dim_expansion,
                                        feedfw_inner_dim_expansion=feedfw_inner_dim_expansion,
                                        attn_use_glu_head=attn_use_glu_head,
                                        attn_use_glu_out=attn_use_glu_out,
                                        feedfw_use_glu=feedfw_use_glu,
                                        dropout_in=dropout_hl_in,
                                        dropout_attn_intern=dropout_attn_intern,
                                        dropout_attn_exit=dropout_attn_exit,
                                        dropout_feedfw=dropout_feedfw,
                                        dropout_out=dropout_hl_out,
                                        depth=i,
                                        initlz_pass_branch=((not self.return_sequences) and i+1==self.num_layers),
                                        add_pos=(add_pos_encoding or i==0))
                                for i in range(self.num_layers) ]

    def build(self, input_shape):
        self.d_model = input_shape[-1] #embedding_size
        if not self.return_sequences:
            self.cls_token = self.add_weight(shape=(self.d_model,), dtype=tf.float32, trainable=True)

    @tf.function     
    def call(self, inputs, mask=None, training=False):
        if self.return_sequences:
            out = inputs
        else:
            batch_size = tf.shape(inputs)[0]
            cls_matrix = tf.expand_dims(tf.expand_dims(self.cls_token, 0), 0)
            cls_matrix = tf.tile(cls_matrix, [batch_size, 1, 1])
            out = tf.concat([cls_matrix, inputs], axis=1)
            
            if mask is not None:
                mm = tf.tile(tf.expand_dims([False], 0), [batch_size,1])
                mask = tf.concat([mm,mask], axis=1)

        for i in range(self.num_layers):
            out = self.layers[i](out, mask=mask)

        if self.return_sequences:
            return out
        else:
            return out[:,0,:]

