import json

from pathlib import Path
from pos_tagger.memm import MEMM
from utils.parser import Parser


if __name__ == "__main__":

    opts = Parser.parse()
    opts.features_params["pairs"] = 100
    opts.features_params["unigrams"] = 50
    opts.features_params["bigrams"] = 100
    opts.features_params["trigrams"] = 150
    opts.features_params["prefixes"] = 25
    opts.features_params["suffixes"] = 25
    opts.features_params["prev_w_curr_t"] = 50
    opts.features_params["next_w_curr_t"] = 50
    opts.features_params["indextagfeatures"] = 50
    opts.features_params["indexwordfeatures"] = 50
    opts.features_params["capitaltagfeatures"] = 50
    opts.force = False
    opts.epochs = 500
    opts.post_processing = True
    opts.beam = 2
    opts.dot = 1.5

    root_dir = Path("models").joinpath(opts.name).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    memm = MEMM(opts, root_dir)
    memm.fit(opts=opts)
    _, opts.test_accuracy = memm.predict(opts.test_file, opts.beam)

    w_path = root_dir.joinpath("params.json")
    with open(w_path, "w") as f:
        opts.train_file = opts.train_file.as_posix()
        opts.test_file = opts.test_file.as_posix()
        json.dump(vars(opts), f, indent=4)
