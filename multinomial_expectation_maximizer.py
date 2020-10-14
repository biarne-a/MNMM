import numpy as np
import pandas as pd
from scipy.stats import multinomial, dirichlet


class MultinomialExpectationMaximizer:
    def __init__(self, K, rtol=1e-3, max_iter=100, restarts=10):
        self._K = K
        self._rtol = rtol
        self._max_iter = max_iter
        self._restarts = restarts

    def compute_log_likelihood(self, X_test, alpha, beta):
        mn_probs = np.zeros(X_test.shape[0])
        for k in range(beta.shape[0]):
            mn_probs_k = self._get_mixture_weight(alpha, k) * self._multinomial_prob(X_test, beta[k])
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()

    def compute_bic(self, X_test, alpha, beta, log_likelihood=None):
        if log_likelihood is None:
            log_likelihood = self.compute_log_likelihood(X_test, alpha, beta)
        N = X_test.shape[0]
        return np.log(N) * (alpha.size + beta.size) - 2 * log_likelihood

    def compute_icl_bic(self, bic, gamma):
        classification_entropy = -(np.log(gamma.max(axis=1))).sum()
        return bic - classification_entropy

    def _multinomial_prob(self, counts, beta, log=False):
        """
        Evaluates the multinomial probability for a given vector of counts
        counts: (N x C), matrix of counts
        beta: (C), vector of multinomial parameters for a specific cluster k
        Returns:
        p: (N), scalar values for the probabilities of observing each count vector given the beta parameters
        """
        n = counts.sum(axis=-1)
        m = multinomial(n, beta)
        if log:
            return m.logpmf(counts)
        return m.pmf(counts)

    def _e_step(self, X, alpha, beta):
        """
        Performs E-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        alpha: (K) or (NxK) in the case of individual weights, mixture component weights
        beta: (K x C), multinomial categories weights

        Returns:
        gamma: (N x K), posterior probabilities for objects clusters assignments
        """
        # Compute gamma
        N = X.shape[0]
        K = beta.shape[0]
        weighted_multi_prob = np.zeros((N, K))
        for k in range(K):
            weighted_multi_prob[:, k] = self._get_mixture_weight(alpha, k) * self._multinomial_prob(X, beta[k])

        # To avoid division by 0
        weighted_multi_prob[weighted_multi_prob == 0] = np.finfo(float).eps

        denum = weighted_multi_prob.sum(axis=1)
        gamma = weighted_multi_prob / denum.reshape(-1, 1)

        return gamma

    def _get_mixture_weight(self, alpha, k):
        return alpha[k]

    def _m_step(self, X, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        gamma: (N x K), posterior probabilities for objects clusters assignments

        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha = self._m_step_alpha(gamma)

        # Compute beta
        beta = self._m_step_beta(X, gamma)

        return alpha, beta

    def _m_step_alpha(self, gamma):
        alpha = gamma.sum(axis=0) / gamma.sum()
        return alpha

    def _m_step_beta(self, X, gamma):
        weighted_counts = gamma.T.dot(X)
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)
        return beta

    def _compute_vlb(self, X, alpha, beta, gamma):
        """
        Computes the variational lower bound
        X: (N x C), data points
        alpha: (K) or (NxK) with individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        gamma: (N x K), posterior probabilities for objects clusters assignments

        Returns value of variational lower bound
        """
        loss = 0
        for k in range(beta.shape[0]):
            weights = gamma[:, k]
            loss += np.sum(weights * (np.log(self._get_mixture_weight(alpha, k)) + self._multinomial_prob(X, beta[k], log=True)))
            loss -= np.sum(weights * np.log(weights))
        return loss

    def _init_params(self, X):
        C = X.shape[1]
        weights = np.random.randint(1, 20, self._K)
        alpha = weights / weights.sum()
        beta = dirichlet.rvs([2 * C] * C, self._K)
        return alpha, beta

    def _train_once(self, X):
        '''
        Runs one full cycle of the EM algorithm

        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        loss = float('inf')
        alpha, beta = self._init_params(X)
        gamma = None

        for it in range(self._max_iter):
            prev_loss = loss
            gamma = self._e_step(X, alpha, beta)
            alpha, beta = self._m_step(X, gamma)
            loss = self._compute_vlb(X, alpha, beta, gamma)
            print('Loss: %f' % loss)
            if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
                    break
        return alpha, beta, gamma, loss

    def fit(self, X):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.

        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        best_loss = -float('inf')
        best_alpha = None
        best_beta = None
        best_gamma = None

        for it in range(self._restarts):
            print('iteration %i' % it)
            alpha, beta, gamma, loss = self._train_once(X)
            if loss > best_loss:
                print('better loss on iteration %i: %.10f' % (it, loss))
                best_loss = loss
                best_alpha = alpha
                best_beta = beta
                best_gamma = gamma

        return best_loss, best_alpha, best_beta, best_gamma


class IndividualMultinomialExpectationMaximizer(MultinomialExpectationMaximizer):
    def __init__(self, K, alpha_init, beta_init, household_ids, rtol=1e-3, max_iter=100, restarts=10):
        super().__init__(K, rtol, max_iter, restarts)
        self._household_ids = household_ids
        self._alpha_init = alpha_init
        self._beta_init = beta_init
        self._household_freqs = np.unique(household_ids, return_counts=True)[1]

    def _init_params(self, X):
        N = X.shape[0]
        alpha = np.vstack([self._alpha_init] * N)
        return alpha, self._beta_init

    def _get_mixture_weight(self, alpha, k):
        return alpha[:, k]

    def _m_step_alpha(self, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        X: (N x C), data points
        gamma: (N x K), probabilities of clusters for objects

        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        gamma_df = pd.DataFrame(gamma, index=self._household_ids)
        grouped_gamma_sum = gamma_df.groupby(gamma_df.index).apply(sum)
        alpha = grouped_gamma_sum.values / grouped_gamma_sum.sum(axis=1).values.reshape(-1, 1)
        alpha = alpha.repeat(self._household_freqs, axis=0)
        return alpha
