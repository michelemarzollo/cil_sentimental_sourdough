#!/usr/bin/env python3
from scipy.sparse import *  # this script needs scipy >= v0.15
import numpy as np
import pickle
import os.path

from utils import print_header_str, print_progress_bar, count_file_lines
from embeddings.config_embeddings import *
from config_shared import ROOT_DIR, vocab_dir, tweet_dir, reuse_computed,verbose

def cooc():
    """Computes GloVe cooccurrence matrix given a vocabulary and the pos. and neg. corpora.
    Entries in the cooccurrence matrix are weighted by the inverse of the distance of the two words.
    # Configs
        :dataset_version        - choose preprocessing
        :emb_dataset            - choose full or small dataset
        :emb_context_window     - context window size
        :emb_word_min_count     - minimum word count for a word to appear in vocab
    """
    if verbose > 0:
        print_header_str('COOCCURRENCES')
    if reuse_computed and os.path.isfile(vocab_dir+cooc_file+'.pkl'):
        if verbose > 0:
            print('Reusing cooccurrence matrix:', cooc_file)
            print_header_str('DONE')
            print()
        return

    with open(vocab_dir+vocab_file+'.pkl', 'rb') as f:
        vocab = pickle.load(f)

    cooc_dict = dict()
    counter = 0

    tot = (count_file_lines(tweet_dir + emb_train_tweets_pos) +
            count_file_lines(tweet_dir + emb_train_tweets_neg) +
            count_file_lines(tweet_dir + emb_test_tweets))
    
    if verbose == 1:
        print_progress_bar(0, tot, prefix = 'Building cooccurrence matrix:', suffix = 'Complete')

    for fn in [tweet_dir + emb_train_tweets_pos, tweet_dir + emb_train_tweets_neg, tweet_dir + emb_test_tweets]:
        with open(fn) as f:
            for line in f:

                # keeps tokens that are not in vocab for proper window construction
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                
                n = len(tokens)
                for i in range(n):
                    for j in range(max(0,i-emb_context_window),min(n,i+emb_context_window)):
                        if i != j and tokens[i] > 0 and tokens[j] > 0:
                            tok = (tokens[i],tokens[j])
                            cooc_dict[tok] = cooc_dict.get(tok,0)+1/abs(i-j)
                counter += 1
                if verbose == 1 and (counter % 5000 == 0 or counter == tot):
                    print_progress_bar(counter, tot, prefix = 'Building cooccurrence matrix:', suffix = 'Complete')
    
    data = list(cooc_dict.values())
    row = [k1 for k1,k2 in cooc_dict.keys()]
    col = [k2 for k1,k2 in cooc_dict.keys()]

    cooc = coo_matrix((data, (row, col)))

    with open(vocab_dir+cooc_file+'.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    if verbose > 0:
        print("{} nonzero entries.".format(cooc.nnz))
        print_header_str('DONE')
        print()

if __name__ == '__main__':
    cooc()
