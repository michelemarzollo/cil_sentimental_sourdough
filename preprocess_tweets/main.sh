#!/bin/bash

DIRECTORIES=(orig-filt pattern-match spelling lemmatization)

for DIR in ${DIRECTORIES[*]}
do
    if [ ! -d "../twitter-datasets-$DIR" ]; then
        mkdir "../twitter-datasets-$DIR"
    fi
    if find "../twitter-datasets-$DIR"/ -mindepth 1 | read; then
        rm "../twitter-datasets-$DIR"/*
    fi
done

echo "`date` > 0 - Creating all necessary directories DONE"

################################################
# 1. filter duplicate tweets from training set #
################################################
for path in ../twitter-datasets/train*.txt; do
    filename=`basename $path`
    cat ../twitter-datasets/$filename | uniq > ../twitter-datasets-${DIRECTORIES[0]}/$filename #XXX sort ?
done
cp ../twitter-datasets/test_data.txt ../twitter-datasets-${DIRECTORIES[0]}/test_data.txt

# build vocab from original tweets
# Frequencies are used to compute word distribution for hashtag unfolding
cat ../twitter-datasets-${DIRECTORIES[0]}/train_pos_full.txt ../twitter-datasets-${DIRECTORIES[0]}/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_orig_with_freq.txt

python vocab_keep_freq.py vocab_orig_with_freq

if [ $? == 0 ]; then
    echo "`date` > 1 - Filtering duplicates and preliminary vocab DONE"
else
    echo "An error occurred while filtering duplicates. See logs above."
    exit 1
fi

################################################
#             2. unfold hashtags               #
################################################
python hashtags_unfold.py

if [ $? == 0 ]; then
    echo "`date` > 2 - Hashtags unfold DONE"
else
    echo "An error occurred while unfolding hashtags. See logs above."
    exit 1
fi

################################################
#   3. pattern matching (slang, contr. forms)  #
################################################
#python compute_pattern_matching_sh.py
./pattern_matching.sh "../twitter-datasets-${DIRECTORIES[1]}"
./pattern_matching.sh "../twitter-datasets-${DIRECTORIES[1]}" #some slip through the first pass

if [ $? == 0 ]; then
    echo "`date` > 3.1 - Pattern matching on words and punctuation DONE"
else
    echo "An error occurred while performing pattern matching on words. See logs above."
    exit 1
fi

# Pattern matching on smiles and numbers
./pattern_match_special_char.sh "../twitter-datasets-${DIRECTORIES[1]}"
./pattern_match_special_char.sh "../twitter-datasets-${DIRECTORIES[1]}"

if [ $? == 0 ]; then
    echo "`date` > 3.2 - Pattern matching on special characters DONE"
else
    echo "An error occurred while performing pattern matching on special characters. See logs above."
    exit 1
fi

#build vocab from filtered and preprocessed tweets
cat ../twitter-datasets-${DIRECTORIES[1]}/train_pos_full.txt ../twitter-datasets-${DIRECTORIES[1]}/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_pattern_match_with_freq.txt

python vocab_keep_freq.py vocab_pattern_match_with_freq

if [ $? == 0 ]; then
    echo "`date` > 3 - pattern matching DONE"
else
    echo "An error occurred while performing pattern matching. See logs above."
    exit 1
fi

################################################
#                 4. spelling                  #
################################################
python spelling.py
./pattern_matching.sh "../twitter-datasets-${DIRECTORIES[2]}/" #some slip through the first pass

if [ $? == 0 ]; then
    echo "`date` > 4 - Spell-checking DONE"
else
    echo "An error occurred while performing spell-checking. See logs above."
    exit 1
fi

################################################
#              5. Lemmatization                #
################################################
python stemming.py

if [ $? == 0 ]; then
    echo "`date` > 5 - Lemmatization DONE"
else
    echo "An error occurred while performing Lemmatization. See logs above."
    exit 1
fi

# Clean-up
rm vocab*txt
rm *.pkl

# Add test files without ID, required to create embeddings
for DIR in ${DIRECTORIES[*]}
do
    sed 's/[^ ]*,//' ../twitter-datasets-$DIR/test_data.txt > ../twitter-datasets-$DIR/test_data_no_id.txt
done

for DIR in ${DIRECTORIES[*]}
do
    cat ../twitter-datasets-$DIR/train_pos_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../twitter-datasets-$DIR/vocab_freq_pos.txt

    python vocab_keep_freq.py ../twitter-datasets-$DIR/vocab_freq_pos

    rm ../twitter-datasets-$DIR/vocab_freq_pos.txt

    cat ../twitter-datasets-$DIR/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../twitter-datasets-$DIR/vocab_freq_neg.txt

    python vocab_keep_freq.py ../twitter-datasets-$DIR/vocab_freq_neg

    rm ../twitter-datasets-$DIR/vocab_freq_neg.txt
done