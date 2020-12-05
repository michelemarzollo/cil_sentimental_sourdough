#!/usr/bin/env python3
from gensim.models import Word2Vec
import numpy as np
import pickle
import os.path

from utils import normalize_matrix, print_header_str
from config_shared import vocab_dir, tweet_dir, embeddings_dir, reuse_computed, verbose
from embeddings.config_embeddings import *

def word2vec():
    """Computes Word2vec embeddings, retrieving corpus from positive and negative tweet files.
    # Configs
        :dataset_version        - choose preprocessing
        :emb_dataset            - choose full or small dataset
        :embedding_dim          - size of embeddings
        :emb_context_window     - context window size
        :emb_word_min_count     - minimum word count for a word to appear in vocab
    """
    if verbose > 0:
        print_header_str('WORD2VEC')

    if (reuse_computed 
        and os.path.isfile(embeddings_dir+selected_embeddings_file+'.npy')
        and os.path.isfile(vocab_dir+vocab_file+'.pkl')):
        if verbose > 0:
            print('Reusing word2vec vocab:', vocab_file)
            print('Reusing word2vec embeddings:', selected_embeddings_file)
            print_header_str('DONE')
            print()
        return

    dataset=[]

    for fn in [tweet_dir+emb_train_tweets_pos, tweet_dir+emb_train_tweets_neg,tweet_dir+emb_test_tweets]:
        with open(fn) as f:
            for line in f:
                tokens = line.strip().split()
                dataset.append(tokens)
    
    model = Word2Vec(dataset, 
                size=embedding_dim, window=emb_context_window, 
                min_count=emb_word_min_count, workers=6, 
                iter=embedding_epochs, sg=1, compute_loss=True)
    
    X = model.wv.vectors
    if embedding_norm:
        X = normalize_matrix(X)

    np.save(embeddings_dir+selected_embeddings_file, X)

    vocab = dict()
    for idx, line in enumerate(model.wv.vocab):
        vocab[line.strip()] = idx
        
    with open(vocab_dir+vocab_file+'.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    if verbose > 0:
        print('Vocabulary size:', len(vocab))
        print('Training loss:', model.get_latest_training_loss())
        print_header_str('DONE')
    print()

if __name__ == '__main__':
    word2vec()
