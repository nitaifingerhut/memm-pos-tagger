import json
import pickle

from pathlib import Path
from pos_tagger.memm import MEMM
from utils.parser import Parser


if __name__ == "__main__":

    opts = Parser.parse()
    opts.features_params["pairs"] = 250
    opts.features_params["unigrams"] = 50
    opts.features_params["bigrams"] = 250
    opts.features_params["trigrams"] = 500
    opts.features_params["prefixes"] = 100
    opts.features_params["suffixes"] = 100
    opts.features_params["prev_w_curr_t"] = 50
    opts.features_params["next_w_curr_t"] = 50
    opts.features_params["indextagfeatures"] = 0
    opts.features_params["indexwordfeatures"] = 0
    opts.features_params["capitaltagfeatures"] = 0
    opts.force = False
    opts.epochs = 500
    opts.post_processing = True
    opts.beam = 2
    opts.dot = 1.5

    root_dir = Path("models").joinpath(opts.name).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    memm = MEMM(opts, root_dir)
    with open(r"models\05-25_00-42-25\trained_weights_data_1.pkl", 'rb') as f:
        weights = pickle.load(f)
        memm.weights = weights[0]
    _, opts.test_accuracy = memm.predict(opts.test_file, opts.beam)

print("a")