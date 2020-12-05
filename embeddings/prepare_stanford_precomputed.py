import codecs
import csv
import numpy as np
import sys
import pickle
import os.path

from utils import print_header_str, normalize_matrix
from config_shared import tweet_dir,embeddings_dir,vocab_dir,dataset_version,reuse_computed,verbose
from embeddings.config_embeddings import *

def extract_embeddings():
    """Creates a vocabulary and an embedding matrix starting from Stanford GloVe embeddings
    # Configs
        :dataset_version        - choose preprocessing
        :emb_dataset            - choose full or small dataset
        :embedding_dim          - size of embeddings
    """
    if verbose > 0:
        print_header_str('STANFORD GLOVE')

    if (reuse_computed and os.path.isfile(vocab_dir+vocab_file+'.pkl')
            and os.path.isfile(embeddings_dir+selected_embeddings_file+'.npy')):
        if verbose > 0:
            print('Reusing vocabulary:', vocab_file)
            print('Reusing embeddings:', selected_embeddings_file)
            print_header_str('DONE')
            print()
        return

    n=0
    ascii_lines = []

    vocab = dict()

    for fn in [tweet_dir + emb_train_tweets_pos, tweet_dir + emb_train_tweets_neg, tweet_dir + emb_test_tweets]:
        with open(fn,'r') as f:
            for line in f:
                words = line.strip().split()
                for w in words:
                    if w in vocab:
                        vocab[w] +=1
                    else:
                        vocab[w] = 1

    with open(stanford_embedding_file, encoding='utf-8') as f:
        for line in f:
            line = line.split(' ')
            
            if vocab.get(line[0],-1) >= emb_word_min_count:
                ascii_lines.append(line)
                n += 1

    if verbose > 0:
        print('All words in dataset:', len(vocab))
        print('Vocabulary size:',n)

    X = np.zeros((n,embedding_dim))
    w2id = {}

    for i,line in enumerate(ascii_lines):
        w2id[line[0]] = i
        X[i] = np.array(line[1:])
    
    if embedding_norm:
        X = normalize_matrix(X)

    np.save(embeddings_dir+selected_embeddings_file, X)
        
    with open(vocab_dir+vocab_file+'.pkl', 'wb') as f:
        pickle.dump(w2id, f, pickle.HIGHEST_PROTOCOL)
    
    if verbose > 0:
        print_header_str('DONE')
        print()

if __name__ == '__main__':
    extract_embeddings()