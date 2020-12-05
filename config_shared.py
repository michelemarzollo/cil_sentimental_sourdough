import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

reuse_computed = True

dataset_version = 'patternmatch_'  # ['', 'filtered_', 'patternmatch_', 'spelling_', 'lemmatized_']

load_tags = False

verbose = 1     # [0 (no output), 1 (full output), 2 (simple log)]

###############
# DIRECTORIES #
###############
tweet_dir = ROOT_DIR + 'twitter-datasets'
if dataset_version == 'filtered_':
    tweet_dir += '-orig-filt'
elif dataset_version == 'patternmatch_':
    tweet_dir += '-pattern-match'
elif dataset_version == 'spelling_':
    tweet_dir += '-spelling'
elif dataset_version == 'lemmatized_':
    tweet_dir += '-lemmatization'
tweet_dir += '/'

if load_tags and dataset_version != 'lemmatized_':
    assert( False and 'POS tags not available for this dataset.')

embeddings_dir = ROOT_DIR + 'embeddings/data/'
datasets_dir = ROOT_DIR + 'datasets/'
vocab_dir = ROOT_DIR + 'vocab/'

nn_runs = ROOT_DIR + 'nn_runs/'
test_predictions_dir = nn_runs + 'test_predictions/'
test_predictions_prob_dir = nn_runs + 'test_predictions_prob/'
val_predictions_dir = nn_runs + 'val_predictions/'
val_predictions_prob_dir = nn_runs + 'val_predictions_prob/'
val_true_labels_dir = nn_runs + 'val_true_labels/'
val_misclassified_dir = nn_runs + 'val_misclassified/'
misclassified_tweets_dir = nn_runs + 'misclassified_tweets/'
plots_dir = nn_runs + 'nn_plots/'
ens_pred = nn_runs + 'ens_pred/'
