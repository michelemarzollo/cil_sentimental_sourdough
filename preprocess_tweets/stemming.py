from scipy.sparse import *    # this script needs scipy >= v0.15
import numpy as np
import pickle
from collections import Counter

from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

import os
from multiprocessing import Pool

from preproc_config import *

import string

#convert treebank pos tag to wordnet pos tag
treebank2wordnet_pos = {
        'CC':None, # coordin. conjunction (and, but, or)  
        'CD':wordnet.NOUN, # cardinal number (one, two)             
        'DT':None, # determiner (a, the)                    
        'EX':wordnet.ADV, # existential ‘there’ (there)           
        'FW':None, # foreign word (mea culpa)             
        'IN':wordnet.ADV, # preposition/sub-conj (of, in, by)   
        'JJ':wordnet.ADJ, # adjective (yellow)                  
        'JJR':wordnet.ADJ, # adj., comparative (bigger)          
        'JJS':wordnet.ADJ, # adj., superlative (wildest)           
        'LS':None, # list item marker (1, 2, One)          
        'MD':None, # modal (can, should)                    
        'NN':wordnet.NOUN, # noun, sing. or mass (llama)          
        'NNS':wordnet.NOUN, # noun, plural (llamas)                  
        'NNP':wordnet.NOUN, # proper noun, sing. (IBM)              
        'NNPS':wordnet.NOUN, # proper noun, plural (Carolinas)
        'PDT':wordnet.ADJ, # predeterminer (all, both)            
        'POS':None, # possessive ending (’s )               
        'PRP':None, # personal pronoun (I, you, he)     
        'PRP$':None, # possessive pronoun (your, one’s)    
        'RB':wordnet.ADV, # adverb (quickly, never)            
        'RBR':wordnet.ADV, # adverb, comparative (faster)        
        'RBS':wordnet.ADV, # adverb, superlative (fastest)     
        'RP':wordnet.ADJ, # particle (up, off)
        'SYM':None, # symbol (+,%, &)
        'TO':None, # “to” (to)
        'UH':None, # interjection (ah, oops)
        'VB':wordnet.VERB, # verb base form (eat)
        'VBD':wordnet.VERB, # verb past tense (ate)
        'VBG':wordnet.VERB, # verb gerund (eating)
        'VBN':wordnet.VERB, # verb past participle (eaten)
        'VBP':wordnet.VERB, # verb non-3sg pres (eat)
        'VBZ':wordnet.VERB, # verb 3sg pres (eats)
        'WDT':None, # wh-determiner (which, that)
        'WP':None, # wh-pronoun (what, who)
        'WP$':None, # possessive (wh- whose)
        'WRB':None, # wh-adverb (how, where)
        '$':None, #  dollar sign ($)
        '#':None, # pound sign (#)
        '"':None, # quote (‘ or “)
        '(':None, # left parenthesis ([, (, {, <)
        ')':None, # right parenthesis (], ), }, >)
        ',':None, # comma (,)
        '.':None, # sentence-final punc (. ! ?)
        ':':None, # mid-sentence punc (: ; ... – -)
    }

def stem_file(fn):
    """
        Stems words in a file and creates another file that contains the POS tags associated with the original input.
        # Params:
            - fn : filename of the file to perform stemming and lemmatization on.
    """
    lem = WordNetLemmatizer()
    with open(tweet_tmp2_dir + fn, 'r') as fin:
        with open(tweet_final_dir + fn, 'w') as fout:
            with open(tweet_final_dir + os.path.splitext(fn)[0] + '_tags.txt', 'w') as ftag:
                for l in fin:
                    prefix = ''
                    if 'test' in fn:
                        comma = l.find(',')
                        prefix = l[:comma]
                        l = l[comma+1:]
                        assert(prefix.isdigit())
                        prefix += ','

                    ll,tags = '',''
                    
                    # First person pronoun is handled differently to improve verb recognition
                    ws = [w if w != 'i' else 'I' for w in l.strip().split(' ') if len(w) > 0]
                    w_pos = []
                    try:
                        w_pos = pos_tag(ws)
                    except Exception as e:
                        print(e)
                        print(ws)
                        ll=l
                    for w,pos in w_pos:
    #                    s = stem.stem(w)
                        if w in ['i', 'us']:
                            pos = 'PRP'
                        if w in string.punctuation:
                            pos = None
                        if pos not in treebank2wordnet_pos or (w.startswith('<') and w.endswith('>')):
                            wnpos = None
                        else:
                            wnpos = treebank2wordnet_pos[pos]
                        
                        if wnpos:
                            lemma = lem.lemmatize(w, pos=wnpos)
                            if wnpos == wordnet.VERB or wnpos == wordnet.ADJ:
                                tags += pos + ' '
                            else:
                                tags += '<> '
                        else:
                            lemma = lem.lemmatize(w)
                            tags += '<> '
                        ll += lemma.lower()+' '
                    fout.write(prefix+ll.strip()+'\n')
                    ftag.write(prefix+tags.strip()+'\n')
    print(f"\tLemmatization: {fn} DONE")

if __name__ == '__main__':
    fns = []
    print('Starting lemmatization. This might take a while...')
    for fn in os.listdir(tweet_tmp2_dir):
        if not os.path.isfile(os.path.join(tweet_tmp2_dir, fn)):
            continue
        fns.append(fn)
        
    p = Pool(7)
    p.map(stem_file, fns)
