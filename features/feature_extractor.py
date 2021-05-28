import functools
import json
import operator
import random

from collections import Counter
from pathlib import Path


class FeatureExtractor(object):
    def __init__(self):
        """
        searches for features in the corpus.
        :param path: path to a file with the corpus.
        """
        self.dicts = {
            "pairs": {},
            "unigrams": {},
            "bigrams": {},
            "trigrams": {},
            "prefixes": {},
            "suffixes": {},
            "prev_w_curr_t": {},
            "next_w_curr_t": {},
            "indextagfeatures": {},
            "indexwordfeatures": {},
            "capitaltagfeatures": {},
        }

    @staticmethod
    def process(path: Path, force_process: bool = True):
        processed_path = path.with_suffix(".json")
        if not processed_path.exists() or force_process:
            fe = FeatureExtractor.extract_features(path)
            print(
                f"\rFeatureExtractor  |  CORPUS = `{path.name}`  |  SAVING EXTRACTED FEATURES TO `{processed_path.name}`"
            )
            fe.save(processed_path)
        else:
            print(
                f"\rFeatureExtractor  |  CORPUS = `{path.name}`  |  LOADING EXTRACTED FEATURES FROM `{processed_path.name}`"
            )
            fe = FeatureExtractor.load(processed_path)
        return fe

    @classmethod
    def extract_features(cls, path: Path):
        fe = cls()
        num_lines = sum(1 for _ in open(path))
        func = lambda key, *args: dict(functools.reduce(operator.add, map(Counter, [fe.dicts[key], *args])))

        with open(path, "r") as f:

            for i, line in enumerate(f.readlines()):
                pairs = [tuple(s.split("_")) for s in line.split()]
                fe.dicts["pairs"] = func("pairs", Counter(pairs))

                unigrams = list(zip(*pairs))[1]
                fe.dicts["unigrams"] = func("unigrams", Counter(unigrams))

                bigrams = [unigrams[i : i + 2] for i in range(len(unigrams) - 1)]
                fe.dicts["bigrams"] = func("bigrams", Counter(bigrams))

                trigrams = [unigrams[i : i + 3] for i in range(len(unigrams) - 2)]
                fe.dicts["trigrams"] = func("trigrams", Counter(trigrams))

                words = list(zip(*pairs))[0]

                indextagfeatures1 = [(unigrams[0], 0)]
                indexwordfeatures1 = [(words[0], 0)]
                if len(words)>=2:
                    indextagfeatures2 = [(unigrams[1], 1)]
                    indexwordfeatures2 = [(words[1], 1)]
                if len(words)>=3:
                    indextagfeatures3 = [(unigrams[2], 2)]
                    indexwordfeatures3 = [(words[2], 2)]
                fe.dicts["indextagfeatures"] = func("indextagfeatures", Counter(indextagfeatures1), Counter(indextagfeatures2), Counter(indextagfeatures3))
                fe.dicts["indexwordfeatures"] = func("indexwordfeatures", Counter(indexwordfeatures1), Counter(indexwordfeatures2), Counter(indexwordfeatures3))


                capital_tag_features = [(w, t) for w, t in zip(words, unigrams)]
                fe.dicts["capitaltagfeatures"] = func("capitaltagfeatures", Counter(capital_tag_features))

                prefixes2 = [w[:2] for w in words if len(w) >= 5]
                prefixes3 = [w[:3] for w in words if len(w) >= 6]
                prefixes4 = [w[:4] for w in words if len(w) >= 7]

                fe.dicts["prefixes"] = func("prefixes", Counter(prefixes2), Counter(prefixes3), Counter(prefixes4))

                suffixes2 = [w[-2:] for w in words if len(w) >= 5]
                suffixes3 = [w[-3:] for w in words if len(w) >= 6]
                suffixes4 = [w[-4:] for w in words if len(w) >= 7]
                fe.dicts["suffixes"] = func("suffixes", Counter(suffixes2), Counter(suffixes3), Counter(suffixes4))

                prev_w_curr_t = [(w, t) for w, t in zip(words[:-1], unigrams[1:])]
                fe.dicts["prev_w_curr_t"] = func("prev_w_curr_t", Counter(prev_w_curr_t))

                next_w_curr_t = [(w, t) for w, t in zip(words[1:], unigrams[:-1])]
                fe.dicts["next_w_curr_t"] = func("next_w_curr_t", Counter(next_w_curr_t))

                print(
                    f"\rFeatureExtractor  |  CORPUS = `{path.name}`  |  EXTRACTING FEATURES [{round(100. * (i + 1) / num_lines, 2)}%%]",
                    end="",
                )
            print()

        # sort each dictionary by frequency
        sort_func = lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
        for key, val in fe.dicts.items():
            val = sort_func(val)
            fe.dicts[key] = val

        return fe

    def filter(self, **kwargs, dot):
        """
        filter top frequent features in each dictionary.
        """
        for arg, val in kwargs.items():
            if arg not in self.dicts.keys():
                pass
            l = len(list(self.dicts[arg].keys()))
            temp = list(self.dicts[arg].keys())[:min(val*dot,l)]
            random.shuffle(temp)
            self.dicts["f_" + arg] = temp[:val]

    def save(self, path: Path):
        data = {}
        for d in self.dicts.keys():
            d_data = []
            for key, val in self.dicts[d].items():
                d_data.append({"key": key, "val": val})
            data[d] = d_data

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: Path):
        with open(path, "r") as f:
            data = json.load(f)
        corpus = cls()

        for d in data.keys():
            d_data = {}
            for v in data[d]:
                k = tuple(v["key"]) if isinstance(v["key"], list) else v["key"]
                d_data[k] = v["val"]
            corpus.dicts[d] = d_data

        return corpus
