import pickle
import utils
import os.path

from embeddings.config_embeddings import *
from config_shared import tweet_dir, vocab_dir,reuse_computed, verbose
from utils import run_script, print_header_str

def create_vocab():
    """Extracts GloVe vocabulary given the positive and negative corpora.
    # Configs
        :dataset_version        - choose preprocessing
        :emb_dataset            - choose full or small dataset
        :emb_word_min_count     - minimum word count for a word to appear in vocab
    """
    if verbose > 0:
        print_header_str('VOCABULARY')
    if reuse_computed and os.path.isfile(vocab_dir+vocab_file+'.pkl'):
        if verbose > 0:
            print('Reusing vocabulary:', vocab_file)
            print_header_str('DONE')
            print()
        return
    
    vocab_filename = vocab_file+'.txt'
    vocab_filename_cut = vocab_file+'_cut.txt'

    # String format for SED argument (only keep word that appear at least min_count times)
    cut_str = '\\({}\\)'.format(
        ''.join([str(i)+('\\|' if i < emb_word_min_count-1 else '') for i in range(0,emb_word_min_count)])
    )
    command_create_vocab = "cat {} {} {} | sed \"s/ /\\n/g\" | grep -v \"^\\s*$\" | sort | uniq -c > {}".format(
            tweet_dir+emb_train_tweets_pos,
            tweet_dir+emb_train_tweets_neg,
            tweet_dir+emb_test_tweets,
            vocab_dir+vocab_filename
        )
    command_cut_vocab = "cat {} | sed \"s/^\\s\\+//g\" | sort -rn | grep -v \"^{}\\s\" | cut -d' ' -f2 > {}".format(
        vocab_dir+vocab_filename, cut_str, vocab_dir+vocab_filename_cut)
    
    utils.run_script(command_create_vocab)
    utils.run_script(command_cut_vocab)
            
    vocab = dict()
    with open(vocab_dir+vocab_filename_cut) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(vocab_dir+vocab_file+'.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


    utils.run_script('rm',
        [vocab_dir+vocab_filename, vocab_dir+vocab_filename_cut])

    if verbose > 0:
        print('Size:', len(vocab), 'words')
        print_header_str('DONE')
        print()

if __name__ == '__main__':
    create_vocab()