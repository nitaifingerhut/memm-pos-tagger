import numpy as np

from pathlib import Path
from typing import List, Tuple
from scipy.sparse import vstack
from utils.history import History
from features.features import Features


class DatasetCreator(object):
    def __init__(self, path: Path, features: Features, ds_tags: List[str], ngram: int = 3):
        super().__init__()

        self.path = path
        self.num_lines = sum(1 for _ in open(self.path))

        self.features = features
        self.n_features = len(features)

        self.ds_tags = ds_tags
        self.n_ds_tags = len(ds_tags)

        self.ngram = ngram
        self.offset = ngram - 1

    def to_features(self, tag: str = "REAL", percentage: float = 0.00):
        with open(self.path, "r") as f:

            features = []
            for q, line in enumerate(f.readlines()):
                pairs = [tuple(s.split("_")) for s in line.split()]
                words = self.offset * ("*",) + list(zip(*pairs))[0]
                tags = self.offset * ("*",) + list(zip(*pairs))[1]

                for i in range(self.offset, len(words)):
                    curr_tags = (
                        tags[(i - self.offset) : i + 1] if tag == "REAL" else tags[(i - self.offset) : i] + (tag,)
                    )
                    history = History(words, curr_tags, i)
                    features.append(self.features.to_vec(history))

                print(
                    f"DatasetCreator  |  CORPUS = `{self.path.name}` [{round(percentage, 2):<5}%%]  |  TAG = {tag:<4} [{round(100. * (q + 1) / self.num_lines, 2)}%%]\r",
                    end="",
                )
            print()

        return vstack(features)

    def process(self):
        ds = []
        for i, t in enumerate(self.ds_tags):
            ds.append(self.to_features(t, 100 * i / self.n_ds_tags))
        print(f"DatasetCreator  |  CORPUS = `{self.path.name}` [{round(100.00, 2):<5}%%]")
        return ds

    def __str__(self) -> str:
        return f"Dataset:: {self.name}"

    def __len__(self) -> int:
        return self.ds.shape[0]

    def __getitem__(self, index) -> np.ndarray:
        return self.ds[index, ...]

    def shape(self) -> Tuple[int, int, int]:
        return self.ds.shape
