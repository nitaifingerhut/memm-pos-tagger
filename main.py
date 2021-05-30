import json

from pathlib import Path
from pos_tagger.memm import MEMM
from utils.parser import Parser


if __name__ == "__main__":

    opts = Parser.parse()

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

    memm.predict_comp("assets/comp1.words", opts.beam)
    memm.predict_comp("assets/comp2.words", opts.beam)
