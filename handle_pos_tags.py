import os

import numpy as np

from config import cls_train_tweets_pos, cls_train_tweets_neg, test_tweets
from config_shared import tweet_dir

pos_to_emb = {
    'JJ': [0, 0],
    'JJR': [0, -1],
    'JJS': [0, 1],
    'VB': [1, 0],   # base form
    'VBD': [-2, 0],  # past tense
    'VBG': [0.5, 0],  # gerund
    'VBN': [-1, 0],  # past participle
    'VBP': [2, 0],  # present tense
    'VBZ': [2, 0],  # present third person
    'empty_pos_tag': [0, 0]
}


def normalize_tags(embeddings):
    """Standardizes the values of pos_to_emb in order to make them have a similar distribution to the one of
    the embeddings to which the values should be appended."""
    means = np.zeros(embeddings.shape[1])
    stds = np.zeros(embeddings.shape[1])
    for i, column in enumerate(embeddings.T):
        means[i] = np.mean(column)
        stds[i] = np.std(column)
    std_emb = np.mean(stds)
    old_dict = np.array(list(pos_to_emb.values()), dtype=float)
    new_dict = None
    for column in old_dict.T:
        max_tag = max(column)
        min_tag = min(column)
        tag_variation = (max_tag - min_tag)  # could also choose (max_tag-min_tag)/2 or np.std(column)
        column = np.reshape((column * std_emb / tag_variation), (-1, 1))
        if new_dict is None:
            new_dict = column
        else:
            new_dict = np.concatenate((new_dict, column), axis=1)
    for i, key in enumerate(pos_to_emb.keys()):
        pos_to_emb[key] = list(new_dict[i])


def extend_vocab_to_tags(vocab, embeddings):
    normalize_tags(embeddings)
    print("Loading POS tags to vocabulary and embedding matrix")
    assert (len(vocab) == embeddings.shape[0])
    empty_tag = np.array([pos_to_emb['empty_pos_tag']] * embeddings.shape[0])
    embeddings = np.append(embeddings, empty_tag, axis=1)

    # np.concatenate is very slow. This is a workaround to perform the op only every 10k rows
    counter = 0
    embeddings_tmp = np.zeros((10000, embeddings.shape[1]))

    for fn in [tweet_dir + cls_train_tweets_pos, tweet_dir + cls_train_tweets_neg, tweet_dir + test_tweets]:
        fn_tags = os.path.splitext(fn)[0] + '_tags.txt'
        with open(fn) as f, open(fn_tags) as f_tags:
            for line, line_tags in zip(f, f_tags):
                for tok, tag in zip(line.strip().split(), line_tags.strip().split()):
                    if tag in pos_to_emb and tok in vocab and tok + tag not in vocab:
                        vocab[tok + tag] = len(vocab)
                        emb_tag = np.array(pos_to_emb[tag])
                        emb_tok_with_tag = embeddings[vocab[tok]]
                        emb_tok_with_tag[-len(pos_to_emb[tag]):] += emb_tag
                        embeddings_tmp[counter] = np.reshape(emb_tok_with_tag, (1, -1))
                        counter += 1
                        if counter % 10000 == 0:
                            embeddings = np.concatenate((embeddings, embeddings_tmp))
                            counter = 0
    embeddings = np.concatenate((embeddings, embeddings_tmp[:counter]))
    assert (len(vocab) == embeddings.shape[0])
    return vocab, embeddings
