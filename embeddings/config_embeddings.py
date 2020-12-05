from config_shared import ROOT_DIR, dataset_version


###########
# ENTROPY #
###########
emb_polar_vocab = 'vocab_freq_{}.pkl'

##############
# EMBEDDINGS #
##############
emb_dataset = '_full'        # ['', '_full']
emb_train_tweets_pos = 'train_pos{}.txt'.format(emb_dataset)
emb_train_tweets_neg = 'train_neg{}.txt'.format(emb_dataset)
emb_test_tweets = 'test_data_no_id.txt'

emb_type = 'glove'      # ['glove','stanford_precomputed','stanford_glove','word2vec']

embedding_dim = 300     # for stanford_precomputed, [25,50,100,200]
embedding_epochs = 50
embedding_norm = False  # Normalization of embeddings

emb_context_window = 10
emb_word_min_count = 10

# Initializa GloVe embeddings with a positivity and negativity factor
glove_polarization = 0

# txt file to fetch Stanford embeddings from
stanford_embedding_file = ROOT_DIR+'glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(embedding_dim)

######################################
# Do not change code below this line #
######################################

if emb_type == 'stanford_precomputed':
    emb_word_min_count = 1

norm = '_normalized' if embedding_norm else ''


def glove_embedding_file_suffix(ep):      #ep=glove_epoch
    return f'{dataset_version}{emb_type}{emb_dataset}_embeddings_wdw{emb_context_window}_mincnt{emb_word_min_count}_{embedding_dim}d_{ep}ep_polar{glove_polarization}{norm}'


if emb_type == 'stanford_precomputed':  # No epochs for these embeddings
    selected_embeddings_file = f'{dataset_version}{emb_type}{emb_dataset}_embeddings_mincnt{emb_word_min_count}_{embedding_dim}d{norm}'
elif emb_type == 'word2vec':
    selected_embeddings_file = f'{dataset_version}{emb_type}{emb_dataset}_embeddings_wdw{emb_context_window}_mincnt{emb_word_min_count}_{embedding_dim}d_{embedding_epochs}ep_{norm}'
elif emb_type == 'stanford_glove':
    selected_embeddings_file = f'{dataset_version}{emb_type}{emb_dataset}_embeddings_wdw{emb_context_window}_mincnt{emb_word_min_count}_{embedding_dim}d_{embedding_epochs}ep_{norm}'
else:
    selected_embeddings_file = glove_embedding_file_suffix(embedding_epochs)

vocab_file = f'vocab_{dataset_version}{emb_type}{emb_dataset}_mincnt{emb_word_min_count}'
cooc_file = f'cooc_{dataset_version}{emb_type}{emb_dataset}_mincnt{emb_word_min_count}_wdw{emb_context_window}'
