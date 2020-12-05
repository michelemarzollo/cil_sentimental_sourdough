import subprocess
import numpy as np
from math import ceil, floor
from config import *
from config_shared import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd


def count_file_lines(filename):
    return len(open(filename).readlines())


def run_script(filename, args=[]):
    command = filename
    for arg in args:
        command += ' ' + arg
    exit_code = subprocess.call(command, shell=True)
    return exit_code


def normalize_matrix(X, ax=0):
    X -= np.mean(X, axis=ax)
    X /= np.std(X, axis=ax)
    return X


def print_header_str(head, tot_width=50):
    center = len(head) + 2  # Make room for spaces
    front = tot_width // 2 - ceil(center / 2)
    back = tot_width // 2 - floor(center / 2)
    print(front * '=' + ' ' + head + ' ' + '=' * back)


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='='):
    """Call in a loop to create terminal progress bar
    # Params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    if filledLength == 0:
        bar = '.' * length
    elif filledLength == length:
        bar = fill * filledLength
    else:
        bar = fill * (filledLength - 1) + '>' + '.' * (length - filledLength)
    # allow for some extra padding at the end for variable line length
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def tweets_statistics():
    """Prints some useful statistics and plots of tweets.
    Only around 0.025% of tweets have more than 40 words (1 tweet in test data)."""
    tweet_lengths = np.array([])

    chosen_tweets = [tweet_dir + cls_train_tweets_pos, tweet_dir + cls_train_tweets_neg]  # [tweet_dir + test_tweets]

    for fn in chosen_tweets:
        with open(fn) as f:
            count = 0
            for line in f:
                tokens = line.strip().split()
                tweet_lengths = np.append(tweet_lengths, len(tokens))
                count += 1

    tweet_lengths = np.sort(tweet_lengths)
    print(tweet_lengths)
    print('total tweets : ' + str(tweet_lengths.size))
    print('Max : ' + str(np.max(tweet_lengths)))
    print('10th bigger : ' + str(tweet_lengths[-10]))
    print('50th bigger : ' + str(tweet_lengths[-50]))
    print('100th bigger : ' + str(tweet_lengths[-100]))
    print('200th bigger : ' + str(tweet_lengths[-200]))
    print('1000th bigger : ' + str(tweet_lengths[-1000]))
    print('Min : ' + str(np.min(tweet_lengths)))
    print('Mean : ' + str(np.mean(tweet_lengths)))
    print('STD : ' + str(np.std(tweet_lengths)))
    plt.hist(tweet_lengths, 50)
    plt.grid(True)
    plt.savefig(ROOT_DIR + 'plots/' + dataset_version + 'tweet_lengths' + ('_train' if len(chosen_tweets) == 2 else '_test'))


def compare_predictions():
    """Makes comparisons between predictions of different classifiers"""
    validation_labels = np.array(pd.read_csv(val_true_labels_dir + dataset_version + 'validation_labels.csv', index_col=0))
    validation_labels = np.reshape(validation_labels, (-1))

    diff_between_files = []
    also1s = []
    also2s = []
    for filename1 in os.listdir(val_predictions_dir):
        if filename1.endswith(".csv"):
            for filename2 in os.listdir(val_predictions_dir):
                if filename2.endswith(".csv"):
                    if filename1 < filename2:
                        wrong1 = 0
                        wrong2 = 0
                        diff_between = 0
                        also1 = 0
                        also2 = 0
                        diff_corr1 = 0
                        diff_corr2 = 0
                        f1 = np.array(pd.read_csv(val_predictions_dir + filename1, index_col=0))
                        f1 = np.reshape(f1, (-1))
                        f2 = np.array(pd.read_csv(val_predictions_dir + filename2, index_col=0))
                        f2 = np.reshape(f2, (-1))
                        for line in range(f1.shape[0]):
                            if f1[line] != validation_labels[line]:
                                wrong1 += 1
                            if f2[line] != validation_labels[line]:
                                wrong2 += 1
                            if f1[line] != f2[line]:
                                diff_between += 1
                                if f1[line] == validation_labels[line]:
                                    diff_corr1 += 1
                                if f2[line] == validation_labels[line]:
                                    diff_corr2 += 1
                            if f1[line] != validation_labels[line]:
                                if f2[line] != validation_labels[line]:
                                    also2 += 1
                            if f2[line] != validation_labels[line]:
                                if f1[line] != validation_labels[line]:
                                    also1 += 1

                        diff_between_files.append(diff_between)
                        print(filename1)
                        print('Wrongly predicted by 1: ' + str(100 * wrong1 / f1.shape[0]) + '%')
                        print(filename2)
                        print('Wrongly predicted by 2: ' + str(100 * wrong2 / f1.shape[0]) + '%')
                        print()
                        print('Differences between files: ' + str(100 * diff_between / f1.shape[0]) + '%')
                        print(f'\t of which correct by 1 {100 * diff_corr1 / diff_between}%, by 2 {100 * diff_corr2 / diff_between}%')
                        also1s.append(also1 / wrong2)
                        also2s.append(also2 / wrong1)
                        print('Wrongly predicted by other among wrong ones: ' + str(100 * also2 / wrong1) + '%, ' + str(
                            100 * also1 / wrong2) + '%\n\n\n')

    print('Max, min and avg differences between files:')
    print(str(100 * max(diff_between_files) / validation_labels.shape[0]) + '%')
    print(str(100 * min(diff_between_files) / validation_labels.shape[0]) + '%')
    print(str(100 * np.mean(diff_between_files) / validation_labels.shape[0]) + '%')

    print('\nWrongly predicted by first that were also wrongly predicted by second:')
    print('Max: ' + str(100 * max(also2s)) + '%')
    print('Min: ' + str(100 * min(also2s)) + '%')
    print('Avg: ' + str(100 * np.mean(also2s)) + '%')

    print('\nWrongly predicted by second that were also wrongly predicted by first:')
    print('Max: ' + str(100 * max(also1s)) + '%')
    print('Min: ' + str(100 * min(also1s)) + '%')
    print('Avg: ' + str(100 * np.mean(also1s)) + '%')

def table2latexcsv(headers, table):
    pre = """\
    \\begin{table}[h!]
    \\begin{center}
    \\caption{Autogenerated table from .csv file.}
    \\label{table1}
    \\pgfplotstabletypeset[
    col sep=comma, % the seperator in our .csv file
    %columns/.style={column type=l},
    display columns/0/.style={
    column name=$$, % name of first column
    column type={S},string type},  % use siunitx for formatting
    """
#      display columns/0/.style={
#		column name=$Value 1$, % name of first column
#		column type={S},string type},  % use siunitx for formatting
#      display columns/1/.style={
#		column name=$Value 2$,
#		column type={S},string type},
    post = """\
    %every head row/.style={
    %before row={\\toprule}, % have a rule at top
    %after row={
    %\\si{\\ampere} & \\si{\\volt}\\ % the units seperated by &
    %\\midrule} % rule under units
    %},
    %every last row/.style={after row=\\bottomrule}, % rule at bottom
    ]{ensemble_models_val_percentage_diff_preds.csv} % filename/path to file
    \\end{center}
    \\end{table}
    """

    latex = pre
    for i,hh in enumerate(headers):
        latex += """\
            display columns/"""+str(i+1)+"""/.style={
            column name="""+hh+""", % name of first column
            column type={S},string type},  % use siunitx for formatting
            """
    latex += post

    llatex = ''
    for l in latex.split('\n'):
        llatex += l.strip(' \t')+'\n'

    csv = ''
    for i,row in enumerate(table):
        l = headers[i]+',' + ','.join([f'{100*x:.2f}\%' if x!=None else '' for x in row])
        csv += l+'\n'
    return llatex, csv

def table2latex(headers, table):
    latex = """\
    \\begin{table}[b]
    \\centering
    \\begin{tabular}{>{\\raggedright\\arraybackslash}m{2.6cm}*{"""+str(len(headers)+1)+"""}{|>{\\centering\\arraybackslash}m{1.3cm}}}
    """
    
    latex += ' & ' + ' & '.join(headers) + '\\\\\n'
    
    latex += """\
    \\hline
    """
    
    for i,row in enumerate(table):
        l = headers[i]+' & ' + ' & '.join([f'{100*x:.2f}\%' if x!=None else '-' for x in row])
        latex += l+'\\\\\n'
    
    latex += """\
    \\end{tabular}
    \\caption{Classification results of different classifiers. The baseline models use Stanford pretrained embeddings, the other ones our implementation of GloVe 300 and pattern-matching.}
    \\label{tab:classification-results}
    \\end{table}"""
    llatex = ''
    for l in latex.split('\n'):
        llatex += l.strip(' \t')+'\n'

    return llatex

def compare_predictions_2latex():
    """Makes comparisons between predictions of different classifiers"""
    validation_labels = np.array(pd.read_csv(val_true_labels_dir + dataset_version + 'validation_labels.csv', index_col=0))
    validation_labels = np.reshape(validation_labels, (-1))

    models = {
                'glove_300_mc10_patternmatch_bat128_stacked_lstm.csv' : ('StackedLSTM',0),
                'glove_300_mc10_patternmatch_bat128_cnn.csv' : ('CNN',1),
                'glove_300_mc10_patternmatch_bat128_double_tcn.csv' : ('DoubleTCN',2),
                'glove_300_mc10_patternmatch_bat128_tcn_lstm.csv' : ('TCN-LSTM',3),
                'glove_300_mc10_patternmatch_bat128_tcn_cnn.csv' : ('TCN-CNN',4),
                'glove_300_mc10_patternmatch_bat128_transformer.csv' : ('Transformer',5),
                'glove_300_mc10_patternmatch_bat128_rnn_attention_lstm512_attention.csv' : ('Attention',6),
#                'glove_300_mc10_patternmatch_bat128_rnn_attention_lstm512_multihead_4_2.csv' : ('',0),
#                'glove_300_mc10_patternmatch_bat128_rnn_attention_lstm512_multihead_16_8.csv' : ('',0),
              }
    
    table_headers = [ None ] * len(models)
    for _, nn in models.items():
        table_headers[nn[1]] = nn[0]
    
    table_diff_preds = [ [None for _ in range(len(models)) ] for _ in range(len(models)) ]
    
    for filename1 in os.listdir(val_predictions_dir):
        if not filename1.endswith(".csv") or filename1 not in models:
            continue
            
        for filename2 in os.listdir(val_predictions_dir):
            if not filename2.endswith(".csv") or filename2 not in models or filename1 >= filename2:
                continue
                
            diff_between, diff_corr1, diff_corr2 = 0, 0, 0
            
            f1 = np.array(pd.read_csv(val_predictions_dir + filename1, index_col=0))
            f1 = np.reshape(f1, (-1))
            f2 = np.array(pd.read_csv(val_predictions_dir + filename2, index_col=0))
            f2 = np.reshape(f2, (-1))
            
            for line in range(f1.shape[0]):
                if f1[line] != f2[line]:
                    diff_between += 1
                    if f1[line] == validation_labels[line]:
                        diff_corr1 += 1
                    if f2[line] == validation_labels[line]:
                        diff_corr2 += 1

            table_diff_preds[models[filename1][1]][models[filename2][1]] = diff_between / f1.shape[0]
            table_diff_preds[models[filename2][1]][models[filename1][1]] = diff_between / f1.shape[0]

#            print(filename1)
#            print(filename2)
#            print()
#            print('Differences between files: ' + str(100 * diff_between / f1.shape[0]) + '%')
#            print(f'\t of which correct by 1 {100 * diff_corr1 / diff_between}%, by 2 {100 * diff_corr2 / diff_between}%')

    latex, csv = table2latexcsv(table_headers,table_diff_preds)
    print(latex)
    print(csv)
    print(""*10)
    print(table2latex(table_headers, table_diff_preds))

def printout_misclassified():
    """Saves to file misclassified tweets to get useful information from them."""
    v = np.array(pd.read_csv(val_misclassified_dir + pred_file_out + cls + '.csv', index_col=None))

    tweets = []
    for fn in [tweet_dir + cls_train_tweets_pos, tweet_dir + cls_train_tweets_neg]:
        print("\n opening" + fn)
        with open(fn) as f:
            for line in f:
                tweets.append(line)

    printout = ['pos: ' + tweets[v[i, 0]] if v[i, 1] == 1 else 'neg: ' + tweets[v[i, 0]] for i in range(v.shape[0])]
    pd.DataFrame(np.array(printout)).replace('\n', '', regex=True).to_csv(
        misclassified_tweets_dir + pred_file_out + cls + '.csv', index=False, sep='\n', header=None)


def misclassified_statistics():

    for fn in os.listdir(misclassified_tweets_dir):      
        tweet_lengths = np.array([])
        print('\n\n' + fn)
        with open(misclassified_tweets_dir + fn) as f:
            count_pos = 0
            count_neg = 0 
            for line in f:
                if line.startswith("pos") or line.startswith("\"pos"):
                    count_pos += 1
                if line.startswith("neg") or line.startswith("\"neg"):
                    count_neg += 1
                tokens = line.strip().split()
                tweet_lengths = np.append(tweet_lengths, len(tokens))

        tweet_lengths = np.sort(tweet_lengths)
        print('Positive: ', count_pos)
        print('Negative: ', count_neg)
        print('Percentage of positives: ', count_pos/(count_pos+count_neg))
        print('total tweets : ' + str(tweet_lengths.size))
        print('Max : ' + str(np.max(tweet_lengths)))
        print('Min : ' + str(np.min(tweet_lengths)))
        print('Mean : ' + str(np.mean(tweet_lengths)))
        print('STD : ' + str(np.std(tweet_lengths)))
        plt.hist(tweet_lengths, 50)
        plt.grid(True)


if __name__ == '__main__':
    # tweets_statistics()
    compare_predictions_2latex()
    # printout_misclassified()
    # misclassified_statistics()
