#!/bin/sh


mkdir datasets
mkdir vocab
cp -R twitter-datasets twitter-datasets-orig
mkdir twitter-datasets-orig-filt
mkdir twitter-datasets-preproc
mkdir twitter-datasets-preprocv4
mkdir twitter-datasets-tmp1
mkdir twitter-datasets-tmp2
mkdir plots

mkdir nn_runs
mkdir nn_runs/nn_plots
mkdir nn_runs/test_predictions
mkdir nn_runs/test_predictions_prob
mkdir nn_runs/val_predictions
mkdir nn_runs/val_predictions_prob
mkdir nn_runs/val_true_labels
mkdir nn_runs/val_misclassified
mkdir nn_runs/misclassified_tweets
mkdir nn_runs/ens_pred
mkdir nn_runs/logs

python leonhard_setup.py install
