#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import os.path
from math import log

from utils import normalize_matrix, print_progress_bar, print_header_str
from embeddings.config_embeddings import *
from config_shared import embeddings_dir, vocab_dir,reuse_computed, verbose, tweet_dir

def sentiment_polarization(vocab_pos, vocab_neg):
    """Computes the polarity of words in the vocabulary based on their relative frequency in the two (positive and negative) datasets.
    # Params
        :vocab_pos  - dictionary that maps words to their frequency in the positive dataset
        :vocab_neg  - dictionary that maps words to their frequency in the negative dataset
    """
    bias = {}

    for w, n_pos in vocab_pos.items():
        n_neg = vocab_neg.get(w,0)
        bias[w] = n_pos/(n_pos+n_neg)
    for w, n_neg in vocab_neg.items():
        if w in bias:
            continue
        bias[w] = 0 # This word is only negative (not in vocab_pos)
    return bias

def glove():
    """Computes GloVe embeddings given a vocabulary and a corresponding cooccurrence matrix.
    # Configs
        :dataset_version        - choose preprocessing
        :emb_dataset            - choose full or small dataset
        :embedding_dim          - size of embeddings
        :emb_context_window     - context window size
        :emb_word_min_count     - minimum word count for a word to appear in vocab
        :glove_polarization     - polarization factor for embedding initialization (with rel. freq)
    """
    if verbose > 0:
        print_header_str('EMBEDDINGS')
    
    if reuse_computed and os.path.isfile(embeddings_dir+selected_embeddings_file+'.npy'):
        if verbose > 0:
            print('Reusing embeddings:', selected_embeddings_file)
            print_header_str('DONE')
            print()
        return

    if verbose > 0:
        print("Loading cooccurrence matrix.")

    with open(vocab_dir+cooc_file+'.pkl', 'rb') as f:
        cooc = pickle.load(f)

    nmax = 100
    
    if verbose > 0:
        print("\tUsing nmax =", nmax, ", with cooc.max() =", cooc.max(),end='\n\n')

        print("Initializing embeddings with U~[-.5,.5] distribution: ", 
            (cooc.shape[0], embedding_dim+1),
            (cooc.shape[1], embedding_dim+1), flush=True, end='\n\n')
    
    xs = np.random.uniform(size=(cooc.shape[0], embedding_dim+1)) - .5
    ys = np.random.uniform(size=(cooc.shape[1], embedding_dim+1)) - .5

    xs /= (embedding_dim+1)
    ys /= (embedding_dim+1)
    # Bias term is incorporated in word embedding
    xs[:,embedding_dim] = 1
    ys[:,embedding_dim-1] = 1
    
    if glove_polarization > 0:
        if verbose > 0:
            print('Adding polarization to random initial embeddings. Factor:', glove_polarization, end='\n\n')
        ### Get bias for positive and negative words ###
        vocab_pos = pickle.load(open(tweet_dir+emb_polar_vocab.format('pos'), 'rb'))
        vocab_neg = pickle.load(open(tweet_dir+emb_polar_vocab.format('neg'), 'rb'))
        polarization = sentiment_polarization(vocab_pos, vocab_neg)

        vocab = pickle.load(open(vocab_dir+vocab_file+'.pkl', 'rb'))

        ############### Add polarization ################
        split = (embedding_dim-1)//2
        for word,id in vocab.items():
            if word in polarization:
                polar = polarization[word]
            else:
                polar = .5
            xs[id,:split] += glove_polarization*polar / (embedding_dim+1)
            xs[id,split:embedding_dim-1] -= glove_polarization*(1-polar) / (embedding_dim+1)
            ys[id,:split] += glove_polarization*polar / (embedding_dim+1)
            ys[id,split:embedding_dim-1] -= glove_polarization*(1-polar) / (embedding_dim+1)
        #################################################
    
    eta = 0.05
    alpha = 3 / 4

    prev_loss = 0.0

    data = [(i,j,n) for i,j,n in zip(cooc.row,cooc.col, cooc.data)]

    for ix, jy, n in data:
            w = min( 1., (n/nmax)**alpha )
            x,y = xs[ix], ys[jy]
            increase_mul = 2*eta*w * ( log(n) - np.dot(x, y) )

            x_upd = xs[ix] + increase_mul*y
            y_upd = ys[jy] + increase_mul*x

            prev_loss += w * ( log(n) - np.dot(x_upd, y_upd) )**2
    
    for epoch in range(embedding_epochs):
        loss = 0.0
        random.shuffle(data)

        if verbose == 1:
            print_progress_bar(0,len(data), prefix='Epoch {:2d}/{:2d}:'.format(epoch+1,embedding_epochs),suffix='- loss difference {:8.2f}'.format(loss-prev_loss))
        counter,missed_updates=0,0
        for ix, jy, n in data:
            counter+=1
            w = min( 1., (n/nmax)**alpha )
            x,y = xs[ix], ys[jy]
            increase_mul = 2*eta*w * ( log(n) - np.dot(x, y) )

            x_upd = xs[ix] + increase_mul*y
            y_upd = ys[jy] + increase_mul*x

            loss_delta = w * ( log(n) - np.dot(x_upd, y_upd) )**2

            # Undo the current update
            if (np.isnan(x_upd).any() or np.isinf(x_upd).any() or
                    np.isnan(y_upd).any() or np.isinf(y_upd).any() or
                    np.isnan(loss+loss_delta) or np.isinf(loss+loss_delta)):
                missed_updates += 1
                loss += w * ( log(n) - np.dot(xs[ix], ys[jy]) )**2
                if (counter % 5000 == 0 or counter == len(data)) and verbose == 1:
                    print_progress_bar(counter,len(data), prefix='Epoch {:2d}/{:2d}:'.format(epoch+1,embedding_epochs),suffix='- loss difference {:8.2f}'.format(loss-prev_loss))
                continue

            xs[ix] = x_upd
            ys[jy] = y_upd

            # Reset bias
            xs[ix,embedding_dim] = 1
            ys[jy,embedding_dim-1] = 1

            loss += loss_delta
            if (counter % 50000 == 0 or counter == len(data)) and verbose == 1:
                print_progress_bar(counter,len(data), prefix='Epoch {:2d}/{:2d}:'.format(epoch+1,embedding_epochs),suffix='- loss difference {:8.2f}'.format(loss-prev_loss))

        ### BOLD DRIVER LEARNING RATE ###
        if prev_loss > loss or epoch==0:
            eta += 0.01*eta
        else:
            eta /= 2
        prev_loss = loss
        #################################

        if verbose > 0:
            print("Epoch {:2d} loss : {:10.2f}".format(epoch+1, loss))
            print('Missed {:4d} updates due to overflow prevention'.format(missed_updates))
            print('Current learning rate: {:1.3f}'.format(eta), end='\n\n', flush=True)

        if (epoch+1) % 10 == 0 and epoch+1 != embedding_epochs:
            X = xs[:,:embedding_dim]
            if embedding_norm:
                X = normalize_matrix(X)
            np.save(embeddings_dir+glove_embedding_file_suffix(epoch+1), X)
    
    # Note: the bias for xs is in position embedding_dim-1
    X = xs[:,:embedding_dim]
    if embedding_norm:
        X = normalize_matrix(X)
    np.save(embeddings_dir+glove_embedding_file_suffix(embedding_epochs), X)

    if verbose > 0:
        print_header_str('DONE')
        print()

if __name__ == '__main__':
    glove()
