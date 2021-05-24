#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw1
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw1) training starts"

# Run the experiments
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 250 --reg-lambda 0.01 --features-params pairs=50 unigrams=50 bigrams=250 trigrams=500 prefixes=50 suffixes=50 prev_w_curr_t=50 next_w_curr_t=50
git add -A
git commit -m "from Newton with love"
git push
