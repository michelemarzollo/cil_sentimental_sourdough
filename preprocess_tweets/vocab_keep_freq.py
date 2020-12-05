#!/usr/bin/env python3
import pickle
import sys

def main():
    '''
    Creates a dictionary to map words in the vocabulary to their frequency in the corpus.
    '''
    vocab = dict()
    with open(sys.argv[1]+'.txt') as f:
        for line in f:
            c, w = line.strip().split(' ')
            vocab[w.strip()] = int(c.strip())

    with open(sys.argv[1].strip('_cut')+'.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()