#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw1
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw1) training starts"

# Run the experiments
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --n-pairs 500 --n-unigrams 500 --n-bigrams 1000 --n-trigrams 1500 --n-prefixes 500 --n-suffixes 500 --n-prev-w-curr-t 500 --n-next-w-curr-t 500 --epochs 10000 --reg-lambda 0.001 --optimizer BatchSGD --batch-size 64
git add -A
git commit -m "auto... BatchSGD"
git push
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --n-pairs 500 --n-unigrams 500 --n-bigrams 1000 --n-trigrams 1500 --n-prefixes 500 --n-suffixes 500 --n-prev-w-curr-t 500 --n-next-w-curr-t 500 --epochs 10000 --reg-lambda 0.001 --optimizer BatchSGD --batch-size 128
git add -A
git commit -m "auto... BatchSGD"
git push
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --n-pairs 500 --n-unigrams 500 --n-bigrams 1000 --n-trigrams 1500 --n-prefixes 500 --n-suffixes 500 --n-prev-w-curr-t 500 --n-next-w-curr-t 500 --epochs 10000 --reg-lambda 0.001 --optimizer BatchSGD --batch-size 256
git add -A
git commit -m "auto... BatchSGD"
git push
python main.py --train-file assets/train1.wtag --test-file assets/test1.wtag --n-pairs 500 --n-unigrams 500 --n-bigrams 1000 --n-trigrams 1500 --n-prefixes 500 --n-suffixes 500 --n-prev-w-curr-t 500 --n-next-w-curr-t 500 --epochs 10000 --reg-lambda 0.001 --optimizer BatchSGD --batch-size 512
git add -A
git commit -m "auto... BatchSGD"
git push