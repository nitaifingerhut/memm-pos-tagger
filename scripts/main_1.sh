#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw1
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw1) training starts"

# Run the experiments
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 10000 --reg-lambda 0.01 --beam 2 --epsilon 1e-12 --name l_0.01_b_2_100 --features-params pairs=1000 unigrams=50 bigrams=1000 trigrams=1500 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250 capital_tag=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 10000 --reg-lambda 0.01 --beam 2 --epsilon 1e-10 --name l_0.01_b_2_101 --features-params pairs=1000 unigrams=50 bigrams=1000 trigrams=1500 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250 capital_tag=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 10000 --reg-lambda 0.01 --beam 3 --epsilon 1e-12 --name l_0.01_b_3_100 --features-params pairs=1000 unigrams=50 bigrams=1000 trigrams=1500 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250 capital_tag=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --epochs 10000 --reg-lambda 0.01 --beam 3 --epsilon 1e-10 --name l_0.01_b_3_101 --features-params pairs=1000 unigrams=50 bigrams=1000 trigrams=1500 prefixes=250 suffixes=250 prev_w_curr_t=250 next_w_curr_t=250 index_tag=250 index_word=250 capital_tag=250
git add -A
git commit -m "from Newton with love"
git push origin daniel-v2