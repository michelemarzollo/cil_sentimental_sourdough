#!/usr/bin/env python3

import embeddings.create_glove_vocab
import embeddings.cooc
import embeddings.glove
import embeddings.word2vec
import embeddings.prepare_stanford_precomputed
import embeddings.stanford_glove
import nns_entry

from config import *

def main():
    if emb_type == 'glove':
        embeddings.create_glove_vocab.create_vocab()
        embeddings.cooc.cooc()
        embeddings.glove.glove()
    elif emb_type == 'word2vec':
        embeddings.word2vec.word2vec()
    elif emb_type == 'stanford_precomputed':
        embeddings.prepare_stanford_precomputed.extract_embeddings()
    elif emb_type == 'stanford_glove':
        embeddings.stanford_glove.stanford_glove()
    else:
        assert(False and 'unrecognized embeddings type')
    
    nns_entry.compute_predictions()

if __name__ == '__main__':
    main()

