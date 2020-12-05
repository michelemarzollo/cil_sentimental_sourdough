# cil_sentimental_sourdough

## Creating required directories and installing dependencies

Running the bash script `setup.sh` will create all the required directories. These either contain structures such as vocabularies and embedding matrices or host prediction files and useful plots.
Moreover, it will install all required dependencies.

Please note that Python &ge; 3.6 is required to run this project.

## Generating the required datasets from `twitter-dataset`

There are 4 preprocessing steps implemented in the directory `preprocess_tweets`, each of which builds on top of the previous one:
- **Filtering** of duplicate tweets in training set
- **Hashtag unfolding** and **pattern matching**: detects hastags and tries to derive the individual words that compose them. Handles some of the most common abbreviations, slang words and genitives; it groups numbers and smiles/emoticons into custom tags.
- **Spell-checking**
- **Lemmatization/spelling**: stems verbs and adjectives to their base forms. This step retains POS tag information in a separate file.

In order to produce the datasets that correspond to each of these steps, you need to install [aspell-python](https://github.com/WojciechMula/aspell-python) (which is already included in this code) by running the following commands:

```bash
$ cd preprocess_tweets/aspell-python
$ python3 setup.3.py build
$ python3 setup.3.py install --user
$ cd .. && ./main.sh
```

Notice that you need to have libaspell headers installed. The Debian package is called `libaspell-dev`, other distributions should have similar package names.

The datasets generated by each step are then contained in an individual directory. 
You can expect this operation to last between 30 and 50 minutes.
Since this computation is unique for all experiments that can be run, we include the preprocessed datasets directly in the repository.

# Configuration

In this project there are 5 configuration files that allow to setup various experiments in terms of the desired datasets, the word embeddings to employ and the algorithms that are used for classifications.

## `config_shared.py`

Sets parameters that are shared along the whole experiment.

| Parameter name | Values | Usage |
| :------------- | :----- | :---- |
| reuse_computed | `Boolean` | If true, vocabularies, cooccurrence matrices and/or embeddings that have already been computed will be loaded from file; otherwise they will be created from scratch every time.    |
| dataset_version | `['','filtered_', 'patternmatch_', 'spelling_', 'lemmatized_']` | Type of preprocessed dataset to use when running an experiment. |
| load_tags | `Boolean` | If true, append a short custom embedding for the Part of Speach (POS) tag associated to a word by the lemmatization step of preprocessing.  |
| verbose | `[0,1,2]` | 0 = no output, 1 = full output (progress bars and auxiliary information), 2 = essential logs only (e.g. at the end of every epoch) |

## `config.py`

Sets parameters that allow to choose for classifiers and classification datasets.

| Parameter name | Values | Usage |
| :------------- | :----- | :---- |
| cls | `['mlp', 'cnn', 'lstm', 'stacked_lstm', 'tcn_lstm', 'double_tcn', 'cnn', 'tcn_cnn', 'rnn_attention', 'transformer']` | Choose a classification algorithm.
| cls_dataset | `['','_full']` | Use small (~200k tweets) or large (~2.2M tweets) dataset resp. for classification. |

## `embeddings/config_embeddings.py`

Sets algorithms, parameters and datasets used to generate word embeddings.

| Parameter name | Values | Algorithm | Usage |
| :------------- | :----- | :---- | :------|
| emb_type | `['glove', 'stanford_precomputed', 'stanford_glove', 'word2vec']` | All | Choose a word embedding algorithm.
| emb_dataset | `['','_full']` | All | Use small (~200k tweets) or large (~2.2M tweets) dataset resp. for classification.
| emb_context_window | `Integer` | GloVe, Word2vec | Size of the context window for each word in the corpus. |
| embedding_dim | `Integer` | All | Size _d_ of a word vector. (NOTE: stanford_precomputed only come in sizes 25, 50, 100 and 200) |
| emb_word_min_count | `Integer` | GloVe, Word2vec, Stanford GloVe | Minimum number of occurrences of a word in the corpus to consider it for the vocabulary.
| embedding_epochs | `Integer` | GloVe, Word2vec, Stanford GloVe | Training epochs for the embedding algorithm. |
| embedding_norm | `Boolean` | All | Controls the normalization of the embedding matrix. | 
| glove_polarization | `Integer` | GloVe | Polarization factor used to bias the initialization of the embedding matrix with a "positivity" and "negativity" score. |

In general, the pipeline can be understood as follows:
- Create a vocabulary based on the min. count of words;
- (GloVe) Create the cooccurrence matrix based on min. count and window size;
- Train embeddings on the selected dataset;
- Save this information on file for further use. Vocabulary files and cooccurrence matrices can be found in the `vocab` directory; embeddings are placed under `embeddings/data`.

**NOTE:** For precomputed GloVe embeddings to work, you need to download the dataset under http://nlp.stanford.edu/data/glove.twitter.27B.zip and unzip it in the root directory of this project (however, the name of the file containing the embeddings may be changed in `embeddings/config_embeddings.py` by adapting the value of `stanford_embedding_file`)

## `neural_networks/nns_config.py`

Allows to change parameters of the different types of classification networks in the project.

## `preprocess_tweets/preproc_config.py`

Sets directories of datasets across Python files at various steps of preprocessing. We do not reccomend changing these filenames, as they might break further steps of the pipeline or preprocessing itelf.

# Running an experiment

Running
```bash
$ python model.py
```
will fetch all current configurations and build the required embedding structures, before proceeding with classification.
