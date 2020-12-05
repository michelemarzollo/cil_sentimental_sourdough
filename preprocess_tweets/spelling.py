from scipy.sparse import *    # this script needs scipy >= v0.15
import numpy as np
import pickle
from collections import Counter

import nltk
import aspell
from math import log

import jellyfish

import os
from multiprocessing import Pool

from preproc_config import tweet_tmp1_dir, tweet_tmp2_dir

slang = [ 'lol', 'afaik', 'afk', 'asap', 'lol', 'lmao', 'rofl', 'roflol',
                'rotflmao', 'wtf', 'irl', 'faq', 'imo', 'imho', 'yey',
                'ok', 'tsk', 'ish', 'yay', 'weed', 'spongebob', 'ahah',
                'eheh', 'muahah', 'buah', 'awwh', 'tumblr' ]

whitelist = [ '<3', '...' ]

MAX_DIST_CORRECTION = 1

def preprocessing():
    """
        Creates wordcost dictionary used to split words that are missing spaces.
    """
    english_dictionary = nltk.corpus.brown.words()
    slang_vocab = pickle.load(open('vocab_pattern_match_with_freq.pkl', 'rb'))

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
    #words = open("words_by_frequency.txt").read().split()
    wordcost = dict((k, log((i+1)*log(len(words_by_freq)))) for i,k in enumerate(words_by_freq))
    maxword = max(len(x) for x in words_by_freq)
    return wordcost,maxword

unfolded = dict()
correct_word = dict()

def spell_file(fn, wordcost, maxword):
    """
        Performs spellchecking using the aspell-python for GNU Aspell.
        If no spelling suggestions are presented, it tries to infer spaces within the word using dynamic programming.
        # Params:
            - fn : filename to perform spellchecking on
            - wordcost : dictionary that associates a cost to each word in the vocabulary
            - maxword : maximum length of a word in the vocabulary
    """

    def infer_spaces(s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""
        global unfolded
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



    speller = aspell.Speller('lang', 'en')
    for w in slang:
        speller.addtoSession(w)
    
    with open(tweet_tmp1_dir + fn, 'r') as fin:
        with open(tweet_tmp2_dir + fn, 'w') as fout:
            res = []
            for l in fin:
                prefix = ''
                if 'test' in fn:
                    comma = l.find(',')
                    prefix = l[:comma].strip()
                    l = l[comma+1:]
                    try:
                        assert(prefix.isdigit())
                    except:
                        print(prefix, l)
                    prefix += ','
                
                ll = ''
                
                ws = [w for w in l.strip().split(' ') if len(w) > 0]
                for w in ws:
                    if w in correct_word:
                        nw = correct_word[w]
                    elif (w.startswith('<') and w.endswith('>')) or w in whitelist or speller.check(w):
                        nw = w
                    else:
                        try:
                            nw1, nw2 = speller.suggest(w)[:2]
                            nwdist1 = jellyfish.levenshtein_distance(w,nw1)
                            nwdist2 = jellyfish.levenshtein_distance(w,nw2)
                            
                            if nw2.count(' ') < nw1.count(' ') or (nwdist1 > MAX_DIST_CORRECTION and nwdist2 < nwdist1) :
                                nw1 = nw2
                                nwdist1 = nwdist2
                            if nwdist1 <= MAX_DIST_CORRECTION:
                                nw = nw1.lower()
                            else:
                                nw = w.lower()
                        except:
                            nw = infer_spaces(w)
                            if nw.count('.') >= nw.count(' ')/3:
                                nw = nw.replace('.', '')
                            elif nw.count('-') >= nw.count(' ')/3:
                                nw = nw.replace('-', '')
                            nw = nw.replace('  ', ' ').lower()
                    ll += nw + ' '
                    correct_word[w] = nw
                res.append(prefix+ll.strip())
#                    fout.write(prefix+ll.strip()+'\n')
            fout.write('\n'.join(res))

if __name__ == '__main__':
    print("Starting spell-check, this might take a while...")
    wordcost,maxword = preprocessing()
    print('\tSpell-checking: preprocessing DONE')
    for fn in os.listdir(tweet_tmp1_dir):
        if not os.path.isfile(os.path.join(tweet_tmp1_dir, fn)):
            continue
        spell_file(fn,wordcost,maxword)
        print('\tSpell-checking: '+fn+' DONE', flush=True)
    #p = Pool(7)
    #p.map(spell_file, fns)
