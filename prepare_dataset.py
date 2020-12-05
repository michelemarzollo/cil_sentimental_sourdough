#!/usr/bin/env python3
from scipy.sparse import *  # this script needs scipy >= v0.15
import numpy as np
import pickle
from math import log, exp, sqrt

from config import *
from config_shared import *
from utils import count_file_lines, print_header_str, print_progress_bar
from handle_pos_tags import pos_to_emb


################### Currently not used #####################

def weight(x, alpha=1):
    return 1  # No weights
    # return 1/(sqrt(x))
    # return 1/(1+exp(-alpha/x))


def entropy(vocab_pos, vocab_neg):
    """Computes entropy of all words in vocabulary.
    # Params
        :vocab_pos  - contains the frequency of all words in positive corpus 
        :vocab_neg  - contains the frequency of all words in negative corpus
    """
    entropy = {}
    min_entr, max_entr = 1.0, 0.0
    for vocab, vocab_other in [(vocab_pos, vocab_neg), (vocab_neg, vocab_pos)]:
        for w in vocab:
            if w in entropy:
                continue

            n1 = vocab[w]
            n2 = 0
            if w in vocab_other:
                n2 = vocab_other[w]

            p1 = n1 / (n1 + n2)
            p2 = n2 / (n1 + n2)
            if p2 == 0:
                p2 = 0.00000001

            entropy[w] = -p1 * log(p1) - p2 * log(p2)

            max_entr = max(max_entr, entropy[w])
            min_entr = min(min_entr, entropy[w])
    global l, r
    l, r = min_entr, max_entr
    if verbose > 0:
        print('Max and min entropies:', max_entr, min_entr)
        print('Max and min weights:', weight(max_entr), weight(min_entr))
        print('Entropy vocabulary size:', len(entropy))

    return entropy


def prob_ws_given_sentiment(vocab):
    n = 0
    for _, c in vocab.items():
        n += c

    prob = {}
    for w, c in vocab.items():
        prob[w] = c / n
    return prob


def salience(vocab_pos, vocab_neg):
    """Computes salience of all words in vocabulary.
    # Params
        :vocab_pos  - contains the frequency of all words in positive corpus 
        :vocab_neg  - contains the frequency of all words in negative corpus
    """
    salience = {}

    prob_pos = prob_ws_given_sentiment(vocab_pos)
    prob_neg = prob_ws_given_sentiment(vocab_neg)

    for prob1, prob2 in [(prob_pos, prob_neg), (prob_neg, prob_pos)]:
        for w in prob1:
            if w in salience:
                continue

            p1 = prob1[w]
            p2 = 0.00000001
            if w in prob2:
                p2 = prob2[w]

            s = 1 - (min(p1, p2) / max(p1, p2))
            #            if s > 0.70:
            salience[w] = s
    print(len(salience))
    # print(salience['i'], salience['<user>'], salience['harvester'])
    return salience


###########################################################################

def get_emb_sum(embeddings, vocab, line):
    tokens = [vocab.get(w, -1) for w in line]
    tokens = [t for t in tokens if t >= 0]

    emb_tweet = np.zeros(embeddings.shape[1], dtype = 'float32')

    for t in tokens:
        emb_tweet += embeddings[t]

    if len(tokens) > 0:
        emb_tweet /= len(tokens)

    return emb_tweet


def compute_dataset_from_embeddings(vocab=None, embeddings=None):
    """Creates matrix of tweet embeddings for baseline. 
    Sums embeddings of words in tweet (possibly weighting the sum with other metrics - entropy, salience...)
    # Configs
        :dataset_version    - choose preprocessing
        :emb_dataset        - choose full or small dataset
        :embedding_dim      - size of embeddings
        :misc, all other configurations are embedding-specific 
            (they have influence on the outcome, but not on the functioning of the module)
    """
    if verbose > 0:
        print_header_str('PREPARE DATASET')

    if vocab is None:
        with open(vocab_dir + vocab_file + '.pkl', 'rb') as f:
            vocab = pickle.load(f)
    if embeddings is None:
        embeddings = np.load(embeddings_dir + selected_embeddings_file + '.npy', allow_pickle=True)

    n_train = count_file_lines(tweet_dir + cls_train_tweets_pos) + count_file_lines(tweet_dir + cls_train_tweets_neg)

    d = embeddings.shape[1]

    x_train = np.zeros((n_train, d), dtype = 'float32')
    y_train = np.zeros(n_train, dtype = 'float32')

    counter = 0

    if verbose == 1:
        print_progress_bar(0, n_train, prefix='Embedding training tweets:')

    for fn in [tweet_dir + cls_train_tweets_pos, tweet_dir + cls_train_tweets_neg]:
        if 'tags' in fn:
            continue

        curr_file_class = 1 if 'pos' in fn else 0

        with open(fn) as f:
            if load_tags:
                fn_tag = os.path.splitext(fn)[0] + '_tags.txt'
                f_tag = open(fn_tag).readlines()
            for line_id, line in enumerate(f):
                line_tag = None
                line = line.strip().split()
                if load_tags:
                    line_tag = f_tag[line_id].strip().split()
                    line = [tok + (tag if tag in pos_to_emb else '') for tok, tag in zip(line, line_tag)]
                x_train[counter] = get_emb_sum(embeddings, vocab, line)
                y_train[counter] = curr_file_class
                counter += 1
                if verbose == 1 and (counter % 5000 == 0 or counter == n_train):
                    print_progress_bar(counter, n_train, prefix='Embedding training tweets:')

    n_test = count_file_lines(tweet_dir + test_tweets)
    x_test = np.zeros((n_test, d), dtype = 'float32')
    counter = 0

    if verbose == 1:
        print_progress_bar(0, n_train, prefix='Embedding test tweets:    ')
    for fn in [tweet_dir + test_tweets]:
        with open(fn) as f:
            if load_tags:
                fn_tag = os.path.splitext(fn)[0] + '_tags.txt'
                f_tag = open(fn_tag).readlines()
            for line_id, line in enumerate(f):
                line = (''.join(line.split(',')[1:])).strip().split()
                line_tag = None
                if load_tags:
                    line_tag = (''.join(f_tag[line_id].split(',')[1:])).strip().split()
                    line = [tok + (tag if tag in pos_to_emb else '') for tok, tag in zip(line, line_tag)]
                x_test[counter] = get_emb_sum(embeddings, vocab, line)

                counter += 1
                if verbose == 1 and (counter % 1000 == 0 or counter == n_test):
                    print_progress_bar(counter, n_test, prefix='Embedding test tweets:    ')

    if verbose > 0:
        print_header_str('DONE')
        print()

    return x_train, y_train, x_test


if __name__ == '__main__':
    compute_dataset_from_embeddings()
