import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.keras import initializers

"""
Implementation of a layer, generating a token/embedding representing the whole sequence
 - summing the embeddings of the words in the sequence
 - weighting differently each word in the corpus (with a learned weight)
"""
class SumEmbeddings(Layer):
    def __init__(self, embeddings, input_length, trainable_embeddings=False):
        super(SumEmbeddings, self).__init__()
        self.vocab_size = embeddings.shape[0]
        self.d_model = embeddings.shape[1]
        
        self.emb_layer = Embedding(input_dim=self.vocab_size, output_dim=self.d_model,
                                    embeddings_initializer=initializers.Constant(embeddings), input_length=input_length,
                                    trainable=trainable_embeddings, mask_zero=True)

        self.weight_emb_layer = Embedding(input_dim=self.vocab_size, output_dim=1,
                                    embeddings_initializer=initializers.RandomNormal(1, 0.01), input_length=input_length,
                                    trainable=True, mask_zero=True)

    @tf.function
    def call(self, inputs):
        """
        Args:
            inputs (tf.tensor): of shape (batch, seq_len, d_model)
            mask   (tf.tensor): of shape (batch, seq_len)
        Returns:
            a tensor of shape (batch, d_model)
        """
        
        input_embs = self.emb_layer(inputs)
        input_weights = self.weight_emb_layer(inputs)

        out = tf.multiply(input_embs, input_weights)

        mask = self.emb_layer.compute_mask(inputs)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), 2)
            out = tf.multiply(out, mask)

        return tf.reduce_sum(out, axis=1)
