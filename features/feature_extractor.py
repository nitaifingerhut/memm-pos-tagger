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
            "index_tag": {},
            "index_word": {},
            "capital_tag": {},
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
                words = list(zip(*pairs))[0]

                pairs_lower = [(w.lower(), t) for w, t in pairs]  # switch words to lower-case
                fe.dicts["pairs"] = func("pairs", Counter(pairs_lower))

                unigrams = list(zip(*pairs))[1]
                fe.dicts["unigrams"] = func("unigrams", Counter(unigrams))

                bigrams = [unigrams[i : i + 2] for i in range(len(unigrams) - 1)]
                fe.dicts["bigrams"] = func("bigrams", Counter(bigrams))

                trigrams = [unigrams[i : i + 3] for i in range(len(unigrams) - 2)]
                fe.dicts["trigrams"] = func("trigrams", Counter(trigrams))

                prefixes2 = [w[:2] for w in words if len(w) >= 5]
                prefixes2 = [w.lower() for w in prefixes2]
                prefixes3 = [w[:3] for w in words if len(w) >= 6]
                prefixes3 = [w.lower() for w in prefixes3]
                prefixes4 = [w[:4] for w in words if len(w) >= 7]
                prefixes4 = [w.lower() for w in prefixes4]
                fe.dicts["prefixes"] = func("prefixes", Counter(prefixes2), Counter(prefixes3), Counter(prefixes4))

                suffixes2 = [w[-2:] for w in words if len(w) >= 5]
                suffixes2 = [w.lower() for w in suffixes2]
                suffixes3 = [w[-3:] for w in words if len(w) >= 6]
                suffixes3 = [w.lower() for w in suffixes3]
                suffixes4 = [w[-4:] for w in words if len(w) >= 7]
                suffixes4 = [w.lower() for w in suffixes4]
                fe.dicts["suffixes"] = func("suffixes", Counter(suffixes2), Counter(suffixes3), Counter(suffixes4))

                prev_w_curr_t = [(w.lower(), t) for w, t in zip(words[:-1], unigrams[1:])]
                fe.dicts["prev_w_curr_t"] = func("prev_w_curr_t", Counter(prev_w_curr_t))

                next_w_curr_t = [(w.lower(), t) for w, t in zip(words[1:], unigrams[:-1])]
                fe.dicts["next_w_curr_t"] = func("next_w_curr_t", Counter(next_w_curr_t))

                index_tag1 = [(unigrams[0], 0)]
                index_word1 = [(words[0].lower(), 0)]
                if len(words) >= 2:
                    index_tag2 = [(unigrams[1], 1)]
                    index_word2 = [(words[1].lower(), 1)]
                if len(words) >= 3:
                    index_tag3 = [(unigrams[2], 2)]
                    index_word3 = [(words[2].lower(), 2)]
                fe.dicts["index_tag"] = func("index_tag", Counter(index_tag1), Counter(index_tag2), Counter(index_tag3))
                fe.dicts["index_word"] = func(
                    "index_word", Counter(index_word1), Counter(index_word2), Counter(index_word3)
                )

                capital_tag = [(w, t) for w, t in zip(words[1:], unigrams[1:]) if w[0].upper() == w[0] and w[0].isalpha()]
                fe.dicts["capital_tag"] = func("capital_tag", Counter(capital_tag))

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

    def filter(self, **kwargs):
        """
        filter top frequent features in each dictionary.
        """
        for arg, val in kwargs.items():
            if arg not in self.dicts.keys():
                pass
            self.dicts["f_" + arg] = list(self.dicts[arg].keys())[:val]
            # l = len(list(self.dicts[arg].keys()))
            # temp = list(self.dicts[arg].keys())[:min(round(val * 1, 0), l)]
            # random.shuffle(temp)
            # self.dicts["f_" + arg] = temp[:val]

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
