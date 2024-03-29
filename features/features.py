import numpy as np

from scipy.sparse import csr_matrix
from utils.history import History
from features.feature import (
    BiGramFeature,
    Feature,
    FEATURES_DICT,
    PairFeature,
    PrefixFeature,
    SuffixFeature,
    TriGramFeature,
    UniGramFeature,
    IndexTagFeature,
    IndexWordFeature,
    CapitalTagFeature,
)
from typing import Iterator, List


class Features(object):
    @staticmethod
    def from_pairs(pairs):
        features = []
        for word, tag in pairs:
            features.append(PairFeature(word, tag))
        return features

    @staticmethod
    def from_unigrams(unigrams):
        features = []
        for unigram in unigrams:
            features.append(UniGramFeature([unigram]))
        return features

    @staticmethod
    def from_bigrams(bigrams):
        features = []
        for bigram in bigrams:
            features.append(BiGramFeature(bigram))
        return features

    @staticmethod
    def from_trigrams(trigrams):
        features = []
        for trigram in trigrams:
            features.append(TriGramFeature(trigram))
        return features

    @staticmethod
    def from_prefixes(prefixes):
        features = []
        for prefix in prefixes:
            features.append(PrefixFeature(prefix))
        return features

    @staticmethod
    def from_suffixes(suffixes):
        features = []
        for suffix in suffixes:
            features.append(SuffixFeature(suffix))
        return features

    @staticmethod
    def from_prev_w_curr_t(prev_w_curr_t):
        features = []
        for prev_w, curr_t in prev_w_curr_t:
            features.append(PairFeature(prev_w, curr_t))
        return features

    @staticmethod
    def from_next_w_curr_t(next_w_curr_t):
        features = []
        for next_w, curr_t in next_w_curr_t:
            features.append(PairFeature(next_w, curr_t))
        return features

    @staticmethod
    def from_index_tag(index_tags):
        features = []
        for index_tag in index_tags:
            features.append(IndexTagFeature(index_tag))
        return features

    @staticmethod
    def from_index_word(index_words):
        features = []
        for index_word in index_words:
            features.append(IndexWordFeature(index_word))
        return features

    @staticmethod
    def from_capital_tag(capital_tags):
        features = []
        for capital_tag in capital_tags:
            features.append(CapitalTagFeature(capital_tag))
        return features

    def __init__(self):
        self.features = []

    def from_data(self, data):
        self.features.extend(self.from_pairs(data["f_pairs"]))
        self.features.extend(self.from_unigrams(data["f_unigrams"]))
        self.features.extend(self.from_bigrams(data["f_bigrams"]))
        self.features.extend(self.from_trigrams(data["f_trigrams"]))
        self.features.extend(self.from_prefixes(data["f_prefixes"]))
        self.features.extend(self.from_suffixes(data["f_suffixes"]))
        self.features.extend(self.from_prev_w_curr_t(data["f_prev_w_curr_t"]))
        self.features.extend(self.from_next_w_curr_t(data["f_next_w_curr_t"]))
        self.features.extend(self.from_index_tag(data["f_index_tag"]))
        self.features.extend(self.from_index_word(data["f_index_word"]))
        self.features.extend(self.from_capital_tag(data["f_capital_tag"]))
        self.features.extend(list(FEATURES_DICT.values()))

    def __str__(self) -> str:
        return f"Features:: {len(self.features)} features"

    def __iter__(self) -> Iterator[Feature]:
        self.iter = iter(self.features)
        return self.iter

    def __next__(self) -> Feature:
        return next(self.iter)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index) -> Feature:
        return self.features[index]

    def __add__(self, other: List[Feature]) -> List[Feature]:
        return self.features + other

    def append(self, feature: Feature):
        self.features.append(feature)
        return self

    def extend(self, features: List[Feature]):
        self.features.extend(features)
        return self

    def to_vec(self, history: History):
        vec = []
        for f in self.features:
            vec.append(f(history))
        return csr_matrix(vec, dtype=np.uint8)

    def to_vec_np(self, history: History):
        vec = []
        for f in self.features:
            vec.append(f(history))
        return np.asarray(vec).astype(np.uint8)
