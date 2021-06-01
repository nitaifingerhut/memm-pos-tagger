import json
import pickle

from pathlib import Path
from pos_tagger.memm import MEMM
from utils.parser import Parser


OPTIMAL_MODEL_PATH = Path("models/final_2").absolute()


if __name__ == "__main__":

    opts = Parser.predict()
    opts.post_processing = True

    root_dir = Path("models").joinpath(opts.name).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)

    features_path = OPTIMAL_MODEL_PATH.joinpath("params.json")
    with open(features_path, "r") as f:
        data = json.load(f)
    opts.features_params = data["features_params"]

    memm = MEMM(opts, root_dir)

    weights_path = OPTIMAL_MODEL_PATH.joinpath("trained_weights_data_1.pkl")
    with open(weights_path, "rb") as f:
            optimal_params = pickle.load(f)
    memm.weights = optimal_params[0]

    predictions = memm.predict_no_gt(Path("assets/comp1.words").absolute(), opts.beam)
    w_path = Path.cwd().joinpath("comp_m1_302919402.wtag")
    with open(w_path, "w") as f:
        f.writelines("\n".join(predictions))

    predictions = memm.predict_no_gt(Path("assets/comp2.words").absolute(), opts.beam)
    w_path = Path.cwd().joinpath("comp_m2_302919402.wtag")
    with open(w_path, "w") as f:
        f.writelines("\n".join(predictions))
