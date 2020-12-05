from embeddings.config_embeddings import *
from config_shared import load_tags

# Length of sentences for truncating input of neural networks
MAX_LENGTH = 40

VALIDATION_SIZE = 100000

ADD_AVERAGE_EMBEDDING: bool = True

BATCH_SIZE = 128

TRAINABLE_EMBEDDINGS: bool = False

# True if you want to save in file names the description of  model, False if you only want the type of model (cls)
full_name = False

######################################
# CHOICE OF CLASSIFIERS AND DATASETS #
######################################
# 'stacked_lstm' and 'tcn_lstm' can be used with GRUs instead of LSTMs, changing a parameter in nns_config in
# the corresponding models. In case you want to use gru set full_name = True
cls = 'tcn_lstm'  # ['mlp', 'lstm', 'stacked_lstm', 'tcn_lstm', 'double_tcn', 'cnn', 'tcn_cnn', 'rnn_attention', 'transformer', 'baseline_lstm']

cls_dataset = '_full'  # ['', '_full']
cls_train_tweets_pos = 'train_pos{}.txt'.format(cls_dataset)
cls_train_tweets_neg = 'train_neg{}.txt'.format(cls_dataset)

test_tweets = 'test_data.txt'

dataset_file_suffix = f'_{selected_embeddings_file}{cls_dataset}'

pred_file_out = f'{emb_type}_{embedding_dim}_mc{emb_word_min_count}_{dataset_version}'

if load_tags:
    pred_file_out += 'tags_'
# pred_file_out += 'bat' + str(BATCH_SIZE) + '_'
if TRAINABLE_EMBEDDINGS:
    pred_file_out += 'trainemb_'
if not ADD_AVERAGE_EMBEDDING:
    pred_file_out += 'no_avg_'
