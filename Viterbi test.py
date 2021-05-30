import json
import pickle
import numpy as np
from utils.history import History
import math
import tqdm

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
    opts.features_params["prev_w_curr_t"] = 150
    opts.features_params["next_w_curr_t"] = 150
    opts.features_params["index_tag"] = 150
    opts.features_params["index_word"] = 150
    opts.features_params["capital_tag"] = 100
    opts.force = False
    opts.epochs = 500
    opts.post_processing = False
    opts.beam = 2
    opts.dot = 1.5

    root_dir = Path("models").joinpath(opts.name).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    memm = MEMM(opts, root_dir)
    with open(r"models\05-27_00-27-51\trained_weights_data_1.pkl", "rb") as f:
        weights = pickle.load(f)
        memm.weights = weights[0]
    memm.ds_tags = list(memm.corpus.dicts["unigrams"].keys())[:5]
    memm.ds_tags_dict = {i: k for i, k in enumerate(memm.ds_tags)}
    pred, opts.test_accuracy = memm.predict(opts.test_file, opts.beam)
    predictions = []
    predictor = []
    with open(opts.test_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            pairs = [tuple(s.split("_")) for s in line.split()]

            sentence = list(list(zip(*pairs))[0])
            real_tags = list(list(zip(*pairs))[1])
            #forward
            n = len(sentence)
            S = ["*", "*"] + [memm.ds_tags for _ in range(n)] + ["."]
            pi = np.zeros((n, len(memm.ds_tags), len(memm.ds_tags)))
            bp = np.zeros((n, len(memm.ds_tags), len(memm.ds_tags)))
            for k in tqdm.tqdm(range(n)):
                for u in range(len(S[k + 1])):
                    for v in range(len(S[k + 2])):
                        bp_max = 0
                        for t in range(len(S[k])):
                            tags = (S[k][t], S[k+1][u], S[k+2][v])
                            hist = History(sentence, tags, k)
                            f = memm.features.to_vec_np(history=hist)
                            numerator = np.exp(f @ memm.weights)
                            denominator = 0
                            for tag in S[k+2]:
                                tags = (S[k][t], S[k+1][u], tag)
                                hist = History(sentence, tags, k)
                                f = memm.features.to_vec_np(history=hist)
                                denominator +=  np.exp(f @ memm.weights)
                            prob = numerator/denominator
                            if k == 0:
                                if pi[k, u, v] < prob:
                                    pi[k, u, v] = prob
                                    bp[k, u, v] = t
                            else:
                                if pi[k, u, v] < pi[k-1,t,u]*prob:
                                    pi[k, u, v] = pi[k-1,t,u]*prob
                                    bp[k, u, v] = t
            #backward
            t = np.zeros(n).astype(int).tolist()
            ind = np.argmax(pi[-1, :, :])
            t[n - 2] = math.floor(ind / pi[-1, :, :].shape[0])
            t[n - 1] = ind - t[n - 2] * pi[-1, :, :].shape[0]
            for k in range(3, n + 1):
                t[n - k] = bp[n - k + 2, t[n - k + 1], t[n - k + 2]].astype(int)

            pred_tags = tuple([memm.ds_tags_dict[i] for i in t])

            pred_line = " ".join([w + "_" + t for w, t in zip(sentence, pred_tags)])
            predictions.append(pred_line)


    print("dd")