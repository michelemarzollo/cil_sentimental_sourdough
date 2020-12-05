from scipy.sparse import *    # this script needs scipy >= v0.15
import numpy as np
import pickle
from collections import Counter
import nltk

import os
from multiprocessing import Pool

from math import log

from preproc_config import *

def hashtags(fn):
    '''
        Detects hashtags in tweets.
        Tries to split them into separate words by using dynamic programming to find the min-cost split.
    '''
    english_dictionary = nltk.corpus.brown.words()
    slang_vocab = pickle.load(open('vocab_orig_with_freq.pkl', 'rb'))

    normalize_english_dict = len(english_dictionary)
    normalize_slang_vocab = 0
    for w, n in slang_vocab.items():
        normalize_slang_vocab += n

    words = {}
    for w, n in Counter(english_dictionary).items():
        words[w] = n/normalize_english_dict
    
    for w, n in slang_vocab.items():
        if w not in words:
            words[w] = 0.
        words[w] += n/normalize_slang_vocab

    words_by_freq = [w for w,_ in sorted(words.items(), key=lambda x: x[1], reverse=True)]

    # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
    wordcost = dict((k, log((i+1)*log(len(words_by_freq)))) for i,k in enumerate(words_by_freq))
    maxword = max(len(x) for x in words_by_freq)

    unfolded = dict()

    def infer_spaces(s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""
        if s in unfolded:
            return unfolded[s]
        
        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
            return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1,len(s)+1):
            c,k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i>0:
            c,k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])
            i -= k

        unfolded[s] = ' '.join(reversed(out))
        return ' '.join(reversed(out))

    with open(tweet_orig_dir + fn, 'r') as fin:
        with open(tweet_tmp1_dir + fn, 'w') as fout:
            for l in fin:
                #preprocessing
                l = l.lower()
                l = ' '.join(w for w in l.split(' ') if len(w) > 0)
                
                ll = ''
                
                pos = 0
                while True:
                    start = l.find('#', pos) # Still works when tweet=ID,#...
                    if start < 0: # end of string -> no more hashtags
                        ll += l[pos:]
                        break
                    end = l.find(' ', start)
                    
                    if end == start+1: # only #
                        ll += l[pos:start+1]
                        pos = end
                        continue
                    
                    hh = l[start+1 : end].strip().strip('#')
                    ll += l[pos:start] + '<hashtag> ' + infer_spaces(hh)
                    pos = end
                fout.write(ll)

if __name__ == '__main__':
    fns = []
    for fn in os.listdir(tweet_orig_dir):
        if not os.path.isfile(os.path.join(tweet_orig_dir, fn)) or 'sample' in fn:
            continue
        fns.append(fn)
        
    p = Pool(7)
    p.map(hashtags, fns)
