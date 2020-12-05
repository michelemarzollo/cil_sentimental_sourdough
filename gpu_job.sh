#!/bin/sh

bsub -n 1 -W 10:00 -o nn_runs/logs/stacked_lstm.out -R "rusage[mem=8192, ngpus_excl_p=1]" python nns_entry.py  #-R "select[gpu_model0==GeForceGTX1080Ti]"