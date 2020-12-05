import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from config import *
from config_shared import *
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from neural_networks.nns_models import *
from embeddings.config_embeddings import emb_type, embedding_dim
from config_shared import verbose
from handle_pos_tags import pos_to_emb, extend_vocab_to_tags
from numpy.random import seed
seed(11)
tf.random.set_seed(12)


def plot_graphs(history, metric, model_name):
    """Helper function for plotting keras training"""
    plt.clf()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.title(model_name + ' ' + emb_type + ' ' + str(embedding_dim))
    plt.savefig(plots_dir + pred_file_out + model_name + '_' + metric)


# only for MLP
def preprocess_dataset(x_tr, y_tr, x_ts):
    scaler = StandardScaler(with_mean=True, with_std=False)
    # scaler = MinMaxScaler(with_mean=True, with_std=False)
    x_tr = scaler.fit_transform(x_tr)
    x_ts = scaler.transform(x_ts)

    return x_tr, y_tr, x_ts


def get_line_embedding(vocab, line):
    """
    Transforms a list of words in list of numbers corresponding to words in vocab.
    :param vocab: the vocabulary where to get word indices
    :param line: the list of word to be converted in list of numbers
    :return: the list of numbers corresponding to word in vocab
    """
    vocab_size = len(vocab)
    tokens = [vocab.get(t, -1) for t in line]
    if ADD_AVERAGE_EMBEDDING:
        tokens = [vocab_size if t == -1 else t for t in tokens]
    tokens = [t + 1 for t in tokens if t >= 0]  # t+1 because 0 in embeddings will be the padding value
    if len(tokens) == 0:
        tokens = [0]  # adds the value for padding all the sequence
    return tokens


def prepare_dataset(vocab):
    """Prepares the training dataset for neural networks different from MLP. A matrix of integers corresponding
    to word keys in the embedding file is created, and to do so each line is padded or truncated. Labels are adjusted
    to be usable by neural networks.
    If load_tags in config_shared.py is True, grammatical information of words is extracted and appended to the word,
    in order to be adapted to the new vocabulary."""
    integer_sentences = []
    y_train = []
    for fn in [tweet_dir + cls_train_tweets_pos, tweet_dir + cls_train_tweets_neg]:
        curr_file_class = 1 if 'pos' in fn else 0
        with open(fn) as f:
            if load_tags:
                fn_tags = os.path.splitext(fn)[0] + '_tags.txt'
                f_tags = open(fn_tags).readlines()
            for id_line, line in enumerate(f):
                line = line.strip().split()
                if load_tags:
                    line_tags = f_tags[id_line].strip().split()
                    line = [tok + (tag if tag in pos_to_emb else '') for tok, tag in zip(line, line_tags)]

                integer_sentences.append(get_line_embedding(vocab, line))
                y_train.append(curr_file_class)

    # creates a num_of_tweets x 40 2d array of padded or truncated tweets
    padded_sentences = tf.keras.preprocessing.sequence.pad_sequences(integer_sentences,
                                                                     maxlen=MAX_LENGTH,
                                                                     padding='post', truncating='post',
                                                                     value=0)
    return padded_sentences, np.array(y_train)


def prepare_test(vocab):
    """Same as prepare_dataset, but for the test set."""
    integer_sentences = []

    for fn in [tweet_dir + test_tweets]:
        with open(fn) as f:
            if load_tags:
                fn_tags = os.path.splitext(fn)[0] + '_tags.txt'
                f_tags = open(fn_tags).readlines()
            for id_line, line in enumerate(f):
                index = line.index(',') + 1
                line = line[index:].strip().split()  # remove index and comma
                if load_tags:
                    line_tag = f_tags[id_line]
                    index = line_tag.index(',') + 1
                    line_tag = line_tag[index:].strip().split()
                    line = [tok + (tag if tag in pos_to_emb else '') for (tok, tag) in zip(line, line_tag)]
                integer_sentences.append(get_line_embedding(vocab, line))

    padded_sentences = tf.keras.preprocessing.sequence.pad_sequences(integer_sentences,
                                                                     maxlen=MAX_LENGTH,
                                                                     padding='post', truncating='post',
                                                                     value=0)
    return padded_sentences


def compute_predictions():
    """Loads vocabulary and embeddings, makes the validation split if working on the full dataset, handles the
    choice of the classifier and trains it.
    Test predictions are made and saved to file, and in case the whole dataset was used, also validation predictions
    are saved.
    For both test and validation also probabilities of predictions are saved to files, in order to use them for
    the final ensemble.
    Results of training are printed and saved to file."""

    vocab = pickle.load(open(vocab_dir + vocab_file + '.pkl', 'rb'))
    embeddings = np.load(embeddings_dir + selected_embeddings_file + '.npy')

    if load_tags:
        vocab, embeddings = extend_vocab_to_tags(vocab, embeddings)

    x_train, y_train = prepare_dataset(vocab)

    idx = np.array([i for i in range(x_train.shape[0])])
    x_train, y_train, idx = shuffle(x_train, y_train, idx, random_state=4321)

    x_val = x_train[-VALIDATION_SIZE:]
    y_val = y_train[-VALIDATION_SIZE:]
    idx_val = idx[-VALIDATION_SIZE:]
    x_train = x_train[:-VALIDATION_SIZE]
    y_train = y_train[:-VALIDATION_SIZE]

    if ADD_AVERAGE_EMBEDDING:
        # add mean embedding for words which don't have an embedding because aren't in vocab
        average_embedding = np.mean(embeddings, axis=0)
        embeddings = np.append(embeddings, np.reshape(average_embedding, (1, -1)), axis=0)
        if load_tags:
            average_embedding[-len(pos_to_emb['empty_pos_tag']):] = pos_to_emb['empty_pos_tag']
    # add the embedding of the blank spaces for the padding (all zeros) as if it was the first word of vocab
    # blank spaces correspond to the value 0 (beginning of embeddings' file)
    embeddings = np.insert(embeddings, 0, np.zeros((1, embeddings.shape[1])), axis=0)

    # creates labels of correct shape
    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)

    if cls == 'lstm':
        model, model_name = lstm(embeddings)
    elif cls == 'stacked_lstm':
        model, model_name = stacked_lstm(embeddings)
    elif cls == 'tcn_lstm':
        model, model_name = tcn_lstm(embeddings)
    elif cls == 'double_tcn':
        model, model_name = double_tcn(embeddings)
    elif cls == 'cnn':
        model, model_name = cnn(embeddings)
    elif cls == 'tcn_cnn':
        model, model_name = tcn_cnn(embeddings)
    elif cls == 'rnn_attention':
        model, model_name = rnn_attention(embeddings)
    elif cls == 'transformer':
        model, model_name = transformer(embeddings)
    elif cls == 'mlp':
        model, model_name = mlp(embeddings)
    elif cls == 'baseline_lstm':
        model, model_name = baseline_lstm(embeddings)
    else:
        assert (False and 'model not defined')

    es = EarlyStopping(monitor='val_loss', verbose=verbose, patience=4, restore_best_weights=True)
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=2, validation_data=(x_val, y_val),
                        epochs=40, callbacks=[es])

    ###############
    # PREDICTIONS #
    ###############
    x_test = prepare_test(vocab)
    y_test_pred = model.predict(x_test)

    if full_name:
        name = model_name
    else:
        name = cls

    # PRINTING TEST PREDICTIONS TO FILE
    with open(test_predictions_dir + pred_file_out + name + '.csv', 'w') as y_test_out:
        y_test_out.write('Id,Prediction\n')
        for i in range(len(y_test_pred)):
            y_test_out.write('%d,%d\n' % (i + 1, (-1 if y_test_pred[i][0] > 0.5 else 1)))

    # Also save class probability predictions (prob. of first class): useful for stacking
    with open(test_predictions_prob_dir + pred_file_out + name + '.csv', 'w') as y_test_out:
        y_test_out.write('Id,Prediction\n')
        for i in range(len(y_test_pred)):
            y_test_out.write('%d,%f\n' % (i + 1, y_test_pred[i][0]))

    y_val_pred = model.predict(x_val)

    # PRINTING VALIDATION PREDICTIONS TO FILE
    y_val_pred_labels = np.array([])
    with open(val_predictions_dir + pred_file_out + name + '.csv', 'w') as y_val_out:
        y_val_out.write('Id,Prediction\n')
        for i in range(len(y_val_pred)):
            y_val_pred_labels = np.append(y_val_pred_labels, (-1 if y_val_pred[i][0] > 0.5 else 1))
            y_val_out.write('%d,%d\n' % (i + 1, y_val_pred_labels[i]))

    validation_loss = log_loss(y_val, y_val_pred)
    y_val = np.argmax(y_val, axis=1)
    # change labels from 0, 1 to -1, 1
    y_val = [1 if i == 1 else -1 for i in y_val]
    validation_accuracy = accuracy_score(y_val, y_val_pred_labels)
    if verbose > 0:
        print("Model_name: {} {}\n".format(model_name, pred_file_out))
        print("Validation loss: %f" % validation_loss)
        print("Validation accuracy: %f\n" % validation_accuracy)

    # Save validation results in log.txt
    with open(ROOT_DIR + 'log.txt', 'a') as log:
        log.write("Model_name: {} {}\n".format(model_name, pred_file_out))
        log.write("Validation loss: %f\n" % validation_loss)
        log.write("Validation accuracy: %f\n\n" % validation_accuracy)

    # Save class probability predictions (prob. of first class): useful for stacking
    with open(val_predictions_prob_dir + pred_file_out + name + '.csv', 'w') as y_val_out:
        y_val_out.write('Id,Prediction\n')
        for i in range(len(y_val_pred)):
            y_val_out.write('%d,%f\n' % (i + 1, y_val_pred[i][0]))

    # Save true validation labels
    with open(val_true_labels_dir + dataset_version + 'validation_labels' + '.csv', 'w') as y_val_out:
        y_val_out.write('Id,Label\n')
        for i in range(len(y_val)):
            y_val_out.write('%d,%d\n' % (i + 1, y_val[i]))

    # Save misclassified tweets of validation set
    with open(val_misclassified_dir + pred_file_out + cls + '.csv', 'w') as y_val_out:
        y_val_out.write('Id,Label\n')
        for yv, yvl, i in zip(y_val, y_val_pred_labels, idx_val):
            if yv != yvl:
                y_val_out.write(f'{i},{yv}\n')

    if verbose > 0:
        print('All predictions were written to file')

    # Saves plots of the training process, with comparison of training and validation results in loss and accuracy
    for metric in ['accuracy', 'loss']:
        plot_graphs(history, metric, name)

    if verbose > 0:
        print('Plots images saved')


if __name__ == '__main__':
    compute_predictions()
