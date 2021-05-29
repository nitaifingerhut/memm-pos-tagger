#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw1
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw1) training starts"

# Run the experiments
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 5000 --reg-lambda 1 --beam 3 --name mid_l_1_b_3 --features-params pairs=250 unigrams=50 bigrams=500 trigrams=1000 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 5000 --reg-lambda 0.1 --beam 3 --name mid_l_0.1_b_3 --features-params pairs=250 unigrams=50 bigrams=500 trigrams=1000 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 5000 --reg-lambda 0.01 --beam 3 --name mid_l_0.01_b_3 --features-params pairs=250 unigrams=50 bigrams=500 trigrams=1000 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 5000 --reg-lambda 0.001 --beam 3 --name mid_l_0.001_b_3 --features-params pairs=250 unigrams=50 bigrams=500 trigrams=1000 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 5000 --reg-lambda 0.0001 --beam 3 --name mid_l_0.0001_b_3 --features-params pairs=250 unigrams=50 bigrams=500 trigrams=1000 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2