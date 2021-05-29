import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import csr_matrix
from typing import List


class Trainer(object):
    def __init__(
        self, true_features: csr_matrix, list_features: List[csr_matrix], reg_lambda: float, num_features: int
    ):
        super().__init__()

        self.true_features = true_features
        self.list_features = list_features
        self.reg_lambda = reg_lambda
        self.num_features = num_features

    def randomize_weights(self):
        return np.random.normal(size=self.num_features).astype(np.float64)

    def linear_termps(self, weights):
        return self.true_features.dot(weights).sum()

    def exps(self, weights):
        exps = []
        for mat in self.list_features:
            exps.append(np.exp(mat.dot(weights)))
        return exps

    def sum_exps(self, exps):
        sum_of_exps = np.zeros(self.true_features.shape[0]).T
        for exp in exps:
            sum_of_exps += exp
        return sum_of_exps

    @staticmethod
    def normalization_term(sum_of_exps):
        if sum_of_exps[sum_of_exps == 0].shape[0] > 1:
            print(sum_of_exps)
            print(np.log(sum_of_exps).sum())
        return np.log(sum_of_exps).sum()

    def regularization_term(self, weights):
        return 0.5 * self.reg_lambda * (weights ** 2).sum()

    def empirical_counts(self):
        return self.true_features.sum(axis=0)

    def calculate_expected_counts(self, exps, sum_of_exps):
        sum_of_mats = csr_matrix(self.true_features.shape).T
        for mat, exp in zip(self.list_features, exps):
            sum_of_mats += csr_matrix.multiply(mat.T, exp)
        return (csr_matrix.multiply(sum_of_mats, (1 / sum_of_exps))).sum(axis=1).T

    def regularization_grad(self, weights):
        return self.reg_lambda * weights

    def calc_objective_per_iter(self, weights):
        exps = self.exps(weights)
        sum_of_exps = self.sum_exps(exps)

        linear_term = self.linear_termps(weights)
        normalization_term = self.normalization_term(sum_of_exps)
        regularization_term = self.regularization_term(weights)
        likelihood = linear_term - normalization_term - regularization_term

        empirical_counts = self.empirical_counts()
        expected_counts = self.calculate_expected_counts(exps, sum_of_exps)
        regularization_grad = self.regularization_grad(weights)
        grad = empirical_counts - expected_counts - regularization_grad

        return (-1) * likelihood, (-1) * grad

    def optimize(self, w_0: np.ndarray = None, epochs: int = 1000, print_every: int = 50):
        if w_0 is None:
            w_0 = self.randomize_weights()
        optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, maxiter=epochs, iprint=print_every)
        return optimal_params
