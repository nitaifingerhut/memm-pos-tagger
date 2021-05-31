import math
import numpy as np
import warnings

from features.features import Features
from typing import Dict, List, Tuple
from utils.history import History
from utils.general import is_digit, NUMBERS, PERSONAL_PRONOUNS, POSSESSIVE_PRONOUNS, PUNCTUATION


class Predictor(object):
    def __init__(self, weights: np.ndarray, features: Features, tags_dict: Dict, beam: int = 2):
        super().__init__()
        self.weights = weights
        self.features = features
        self.tags_dict = tags_dict
        self.reverse_tags_dict = {v: k for k, v in tags_dict.items()}
        self.tags = list(self.tags_dict.values())
        self.n_tags = len(self.tags)
        self.beam = beam

        # statistics
        self.confusion = np.zeros(shape=(self.n_tags, self.n_tags), dtype=np.int32)

    def predict(self, sentence: List[str], apply_post_processing: bool = True):
        pred_indices, pi, bp = self.viterbi(sentence)
        pred_tags = tuple([self.tags_dict[i] for i in pred_indices])
        if apply_post_processing:
            pred_tags = self.post_processing(sentence, pred_tags)
        return pred_tags

    def append_stats(self, real, pred):
        real_indices = [self.reverse_tags_dict[t] for t in real]
        pred_indices = [self.reverse_tags_dict[t] for t in pred]
        for real_ind, pred_ind in zip(real_indices, pred_indices):
            self.confusion[real_ind, pred_ind] += 1
            self.confusion[pred_ind, real_ind] += 1

    def get_stats(self, slice: int = 10):
        num_correct = np.trace(self.confusion)
        num_samples = np.sum(self.confusion)
        accuracy = 100.0 * num_correct / num_samples
        return accuracy, self.confusion[:slice, :slice], self.tags[:slice]

    def largest_indices(self, p):
        idx = np.argsort(p.ravel())[: -self.beam - 1 : -1]
        return np.column_stack(np.unravel_index(idx, p.shape))

    def get_prob_base(self, sentence, labels, t_2_labels, t_1_labels, ind, calc_ind):
        """
        calculates the probability for each possible history
        labels - list of all possible labale for the current word
        sentence - current sentence
        t_2_labels - list of all possible labale for the t-2 word
        t_1_labels - list of all possible labale for the t-1 word
        ind - current word index
        calc_ind = boolean matrix for beam search of shape(len(t_2_labels), len(t_1_labels))
        """
        e_f = np.zeros((len(t_2_labels), len(t_1_labels), len(labels)))
        for i in range(len(t_2_labels)):
            for j in range(len(t_1_labels)):
                for q in range(len(labels)):
                    if calc_ind[i, j] == 1:
                        tags = (t_2_labels[i], t_1_labels[j], labels[q])
                        hist = History(sentence, tags, ind)
                        f = self.features.to_vec_np(history=hist)
                        e_f[i, j, q] = np.exp(f @ self.weights)
        e_f_sum = np.sum(e_f, axis=2)

        warnings.simplefilter("ignore")
        prob = np.nan_to_num(e_f / e_f_sum[..., None])
        warnings.simplefilter("default")

        return prob

    def viterbi(self, sentence):
        # forward
        n = len(sentence)
        pi = np.zeros((n, self.n_tags, self.n_tags))
        bp = np.zeros((n, self.n_tags, self.n_tags))
        S = ["*", "*"] + [self.tags for _ in range(n)] + ["."]
        for k in range(n):
            if k == 0:
                # pi[-1] = 1
                prob = self.get_prob_base(sentence, S[k + 2], S[k], S[k + 1], k, np.ones((1, 1)))
                pi[k, 0, :] = np.max(prob, axis=0)
            else:
                prob = self.get_prob_base(sentence, S[k + 2], S[k], S[k + 1], k, to_calc)
                temp = pi[k - 1, :, :, np.newaxis] * prob
                pi[k, :, :] = np.max(temp, axis=0)
                bp[k, :, :] = np.argmax(temp, axis=0)

            to_calc = np.zeros((len(S[k + 1]), len(S[k + 2])))
            if self.beam == 0:
                to_calc += 1
            else:
                c = self.largest_indices(pi[k, :, :])
                to_calc[c[:, 0], c[:, 1]] = 1
        # backward
        t = np.zeros(n).astype(int).tolist()
        # argmax change the metric into vector and then returns the ind of the max
        ind = np.argmax(pi[-1, :, :])
        # calc the row of the argmax
        t[n - 2] = math.floor(ind / pi[-1, :, :].shape[0])
        # calc the column
        t[n - 1] = ind - t[n - 2] * pi[-1, :, :].shape[0]
        for k in range(3, n + 1):
            t[n - k] = bp[n - k + 2, t[n - k + 1], t[n - k + 2]].astype(int)
        return t, pi, bp

    @staticmethod
    def post_processing(sentence: Tuple, pred_tags: Tuple):
        processed_tags = ()
        for w, t in zip(sentence, pred_tags):
            if w in PUNCTUATION.keys():
                t = PUNCTUATION[w]
            if is_digit(w) or w in NUMBERS:
                t = "CD"
            if w.lower() in PERSONAL_PRONOUNS:
                t = "PRP"
            if w.lower() in POSSESSIVE_PRONOUNS:
                t = "PRP$"
            processed_tags += (t,)
        return processed_tags
