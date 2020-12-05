#!/usr/bin/env python3
from gensim.models import Word2Vec
import numpy as np
import pickle
import os.path

from utils import normalize_matrix, print_header_str, run_script
from config_shared import vocab_dir, tweet_dir, embeddings_dir, reuse_computed, verbose
from embeddings.config_embeddings import *

def stanford_glove():
    """Computes Word2vec embeddings, retrieving corpus from positive and negative tweet files.
    # Configs
        :dataset_version        - choose preprocessing
        :emb_dataset            - choose full or small dataset
        :embedding_dim          - size of embeddings
        :emb_context_window     - context window size
        :emb_word_min_count     - minimum word count for a word to appear in vocab
    """
    if verbose > 0:
        print_header_str('STANFORD GLOVE')

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

    stanford_root_dir = embeddings_dir+'../StanfordGloVe/'

    with open(stanford_root_dir+'run.sh', 'w') as frun:
        frun.write(f"""\
#!/bin/bash
set -e

pushd {stanford_root_dir}
make
popd

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

CORPUS="{tweet_dir+emb_train_tweets_pos} {tweet_dir+emb_train_tweets_neg} {tweet_dir+emb_test_tweets}"
VOCAB_FILE={stanford_root_dir+vocab_file}_cnt.txt
COOCCURRENCE_FILE={stanford_root_dir}cooccurrence.bin
COOCCURRENCE_SHUF_FILE={stanford_root_dir}cooccurrence.shuf.bin
BUILDDIR={stanford_root_dir}build
SAVE_FILE={stanford_root_dir+selected_embeddings_file}_tmp
VERBOSE=2
MEMORY=8.0
VOCAB_MIN_COUNT={emb_word_min_count}
VECTOR_SIZE={embedding_dim}
MAX_ITER={embedding_epochs}
WINDOW_SIZE={emb_context_window}
BINARY=2
NUM_THREADS=6
X_MAX=100

echo
echo "$ cat CORPUS | BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE > VOCAB_FILE"
cat $CORPUS | $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE > $VOCAB_FILE

echo "$ cat CORPUS | BUILDDIR/cooccur -memory $MEMORY -vocab-file VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE > COOCCURRENCE_FILE"
cat $CORPUS | $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE > $COOCCURRENCE_FILE

echo "$ BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < COOCCURRENCE_FILE > COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

echo "$ BUILDDIR/glove -save-file SAVE_FILE -threads $NUM_THREADS -input-file COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

rm $COOCCURRENCE_FILE $COOCCURRENCE_SHUF_FILE
        """)

    stanford_glove_cmd = 'chmod +x ' + stanford_root_dir+'run.sh && '
    stanford_glove_cmd += stanford_root_dir+'run.sh'
    run_script(stanford_glove_cmd)

    vocab_size = sum(1 for line in open(stanford_root_dir+selected_embeddings_file+'_tmp.txt', 'r'))
    vocab = {}
    embeddings = np.zeros((vocab_size, embedding_dim), dtype = 'float32')

    with open(stanford_root_dir+selected_embeddings_file+'_tmp.txt', 'r') as f:
        for i,l in enumerate(f):
            ll = l.strip().split(' ')
            word, emb = ll[0].strip(), [float(x.strip()) for x in ll[1:]]
            
            vocab[word] = i
            embeddings[i] = np.array(emb)

    
    if embedding_norm:
        embeddings = normalize_matrix(embeddings)

    np.save(embeddings_dir+selected_embeddings_file, embeddings)
        
    with open(vocab_dir+vocab_file+'.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    cleanup_cmd = f'rm {stanford_root_dir+vocab_file}_cnt.txt {stanford_root_dir+selected_embeddings_file}_tmp.txt ; rm -rf {stanford_root_dir}build'
    run_script(cleanup_cmd)

    if verbose > 0:
        print('Vocabulary size:', len(vocab))
        print_header_str('DONE')
    print()

if __name__ == '__main__':
    stanford_glove()
