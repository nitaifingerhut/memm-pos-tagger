import json
import pickle

from pathlib import Path
from pos_tagger.memm import MEMM
from utils.parser import Parser


INFO = {
    "comp1": {
        "params": Path("models/final_2/params.json").absolute(),
        "weights": Path("models/final_2/trained_weights_data_1.pkl").absolute(),
        "src_path": Path("assets/comp1.words").absolute(),
        "trg_path": Path("comp_m1_302919402.wtag").absolute(),
    },
    "comp2": {
        "params": Path("models/final_2_small/params.json").absolute(),
        "weights": Path("models/final_2_small/trained_weights_data_2.pkl").absolute(),
        "src_path": Path("assets/comp2.words").absolute(),
        "trg_path": Path("comp_m2_302919402.wtag").absolute(),
    }
}


if __name__ == "__main__":

    opts = Parser.predict()
    opts.post_processing = True

    root_dir = Path("models").joinpath(opts.name).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)

    features_path = INFO[opts.comp_file_name]["params"]
    with open(features_path, "r") as f:
        data = json.load(f)
    opts.features_params = data["features_params"]

    memm = MEMM(opts, root_dir)

    weights_path = INFO[opts.comp_file_name]["weights"]
    with open(weights_path, "rb") as f:
        optimal_params = pickle.load(f)
    memm.weights = optimal_params[0]

    predictions = memm.predict_no_gt(INFO[opts.comp_file_name]["src_path"], opts.beam)

    w_path = INFO[opts.comp_file_name]["trg_path"]
    with open(w_path, "w") as f:
        f.writelines("\n".join(predictions))
