"""This file contains all neural network models. For every model the Sequential keras model
is used, by combining either built-in keras layers or custom layers. The name and construction of models
is self-explicative. For further information on custom layers see the corresponding files in this directory.
Configurations of these models can be found in nns_config.py, in this same directory."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.losses import BinaryCrossentropy

from neural_networks.nns_config import *
from neural_networks.temporal_convolution import TCN
from neural_networks.cnn_model import CNN
from neural_networks.residual_rnn import ResidualRNN
from neural_networks.attention import Attention, MultiHead_Attention
from neural_networks.glu import GLU
from neural_networks.transformer import Transformer
from neural_networks.sum_emb_layer import SumEmbeddings
from config import MAX_LENGTH, TRAINABLE_EMBEDDINGS


def baseline_lstm(embeddings: np.ndarray) -> (tf.keras.Model, str):
    model_name = "baseline_lstm"
    config = nns_config['stacked_lstm']

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=True))
    model.add(LSTM(config['lstm_units'][0], dropout=config['lstm_dropout']))
    model.add(Dropout(config['dense_dropout']))
    for units in [256, 256]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def lstm(embeddings: np.ndarray) -> (tf.keras.Model, str):
    model_name = "lstm"
    config = nns_config['stacked_lstm']
    model_name = '{}_{}_dense{}x2'.format(model_name, str(config['lstm_units'][0]), str(config['dense_units'][0]))

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=True))
    model.add(Bidirectional(LSTM(config['lstm_units'][0], dropout=config['lstm_dropout'])))
    model.add(Dropout(config['dense_dropout']))
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def stacked_lstm(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['stacked_lstm']
    model_name = config['model_name']

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=True))
    for i in range(len(config['lstm_units'])):
        model.add(ResidualRNN(rnn_type=config['rnn_type'],
                              units=config['lstm_units'][i],
                              dropout=config['lstm_dropout'],
                              residual=config['residual'],
                              high_init=config['high_init'],
                              return_sequences=(True if i < len(config['lstm_units']) - 1 else False)))
    model.add(Dropout(config['dense_dropout']))
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def tcn_lstm(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['tcn_lstm']
    model_name = config['model_name']

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=config['use_mask']))
    model.add(TCN(num_channels=config['channels'],
                  kernel_size=config['kernel_size'],
                  stride=config['stride'],
                  dropout=config['dropout'],
                  use_highway=config['use_highway'],
                  high_init=config['high_init'],
                  use_glu=config['use_glu'],
                  attention=config['attention'],
                  attention_dropout=config['att_dropout'],
                  keep_len=config['keep_len'],
                  causal=config['causal'],
                  propagate_masking=config['use_mask']))
    model.add(ResidualRNN(rnn_type=config['rnn_type'],
                          units=config['lstm_units'],
                          dropout=config['lstm_dropout'],
                          residual='none',
                          high_init=0,
                          return_sequences=False))
    model.add(Dropout(config['dense_dropout']))
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def double_tcn(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['double_tcn']
    model_name = config['model_name']

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=False))
    model.add(TCN(num_channels=config['channels_1'],
                  kernel_size=config['kernel_size_1'],
                  stride=config['stride_1'],
                  dropout=config['dropout_1'],
                  use_highway=config['use_highway_1'],
                  high_init=config['high_init_1'],
                  use_glu=config['use_glu_1'],
                  attention=config['attention_1'],
                  attention_dropout=config['att_dropout_1'],
                  keep_len=config['keep_len_1'],
                  causal=config['causal_1']))
    model.add(Dropout(config['intermediate_dropout']))
    model.add(TCN(num_channels=config['channels_2'],
                  kernel_size=config['kernel_size_2'],
                  stride=config['stride_2'],
                  dropout=config['dropout_2'],
                  use_highway=config['use_highway_2'],
                  high_init=config['high_init_2'],
                  use_glu=config['use_glu_2'],
                  attention=config['attention_2'],
                  attention_dropout=config['att_dropout_2'],
                  keep_len=config['keep_len_2'],
                  causal=config['causal_2']))
    model.add(Flatten())
    model.add(Dropout(config['dense_dropout']))
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def cnn(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['cnn']
    model_name = config['model_name']

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=False))
    model.add(CNN(filters=config['filters'],
                  kernel_sizes=config['kernel_sizes'],
                  strides=config['strides'],
                  dilation_rates=config['dilations']))
    model.add(Dropout(config['dense_dropout']))
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def tcn_cnn(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['tcn_cnn']
    model_name = config['model_name']

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=False))
    model.add(TCN(num_channels=config['channels'],
                  kernel_size=config['kernel_size'],
                  stride=config['stride'],
                  dropout=config['dropout'],
                  use_highway=config['use_highway'],
                  high_init=config['high_init'],
                  use_glu=config['use_glu'],
                  attention=config['attention'],
                  attention_dropout=config['att_dropout'],
                  keep_len=config['keep_len'],
                  causal=config['causal']))
    model.add(Dropout(config['pre_cnn_dropout']))
    model.add(CNN(filters=config['filters'],
                  kernel_sizes=config['kernel_sizes'],
                  strides=config['strides'],
                  dilation_rates=config['dilations']))
    model.add(Dropout(config['dense_dropout']))
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def rnn_attention(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['rnn_attention']
    model_name = config['model_name']

    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=True))
#    model.add(GRU(config['rnn_units'], return_sequences=(config['attention_mode'] != 'none')))
    model.add(ResidualRNN(rnn_type=config['rnn_type'],
                          units=config['rnn_units'],
                          dropout=config['rnn_in_dropout'],
                          residual=config['rnn_highway_mode'],
                          high_init=config['rnn_high_init'],
                          # use_weight_norm=config['rnn_use_weight_norm'],
                          return_sequences=(config['attention_mode'] != 'none')))

    if config['use_glu']:
        model.add(GLU(axis=1 if config['attention_mode'] == 'none' else 2))

    if config['attention_mode'] == 'attention':
        model.add(Attention())
    elif config['attention_mode'] == 'multihead_attention':
        model.add(MultiHead_Attention(n_head=config['multihead_nheads'], inner_dims_expansion=config['multihead_dim_exp']))
    elif config['attention_mode'] == 'transformer':
        model.add(Transformer(num_layers=config['transformer_nlayers'],
                              n_head=config['transformer_attn_heads'],
                              attn_inner_dim_expansion=config['transformer_attn_dim_exp'],
                              feedfw_inner_dim_expansion=config['transformer_feedfw_dim_exp']))
    elif config['attention_mode'] != 'none':
        assert (False and 'attention mode unknown')

    model.add(Dropout(config['rnn_out_dropout']))

    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def transformer(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['transformer']
    model_name = config['model_name']
    
    model = Sequential()
    model.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],
                        embeddings_initializer=Constant(embeddings),
                        input_length=MAX_LENGTH, trainable=TRAINABLE_EMBEDDINGS, mask_zero=True))
    model.add(Transformer(num_layers=config['transformer_layers'],
                          n_head=config['transformer_attn_heads'],
                          attn_inner_dim_expansion=config['transformer_attn_dim_exp'],
                          feedfw_inner_dim_expansion=config['transformer_feedfw_dim_exp'],
                          attn_use_glu_head=config['transformer_attn_use_glu_head'],
                          attn_use_glu_out=config['transformer_attn_use_glu_out'],
                          feedfw_use_glu=config['transformer_feedfw_use_glu'],
                          dropout_hl_in=config['transformer_drop_hl_in'],
                          dropout_attn_intern=config['transformer_drop_attn_intern'],
                          dropout_attn_exit=config['transformer_drop_attn_exit'],
                          dropout_feedfw=config['transformer_drop_feedfw'],
                          dropout_hl_out=config['transformer_drop_hl_out'],
                          return_sequences=config['transformer_return_sequences'],
                          add_pos_encoding=config['transformer_add_pos_encodings']))

    model.add(Dropout(config['transformer_dropout']))

    #    model.add(Attention())
    #    model.add(Dropout(config['transformer_dropout']))

    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dense_dropout']))
    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=config['lr'])
    model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model, model_name


def mlp(embeddings: np.ndarray) -> (tf.keras.Model, str):
    config = nns_config['mlp']

    model = Sequential()
    model.add(SumEmbeddings(embeddings=embeddings, input_length=MAX_LENGTH, trainable_embeddings=TRAINABLE_EMBEDDINGS))

    for u in config['dense_units']:
        model.add(Dense(u, activation='relu', kernel_regularizer=l2(0.0001)))
        model.add(Dropout(config['dense_dropout']))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
    model.build((None, MAX_LENGTH))
    print(model.summary())
    return model, config['model_name']
