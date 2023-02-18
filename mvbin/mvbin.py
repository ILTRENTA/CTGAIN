"""
Author: Shadi Zabad
Date: March 2021

This is a script that implements functions for generating correlated binary data using
the method described in Leisch et al. (1998).

This software is a python implementation of the algorithm outlined in:

On the Generation of Correlated Artificial Binary Data
Friedrich Leisch, Andreas Weingessel, Kurt Hornik (1998)

Implementation follows closely the implementation details in the R package `bindata`
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy import interpolate
from scipy.stats.mvn import mvnun


def admissible_joint_prob(jp):
    """
    This function checks that the joint probability matrix
    has admissible values.
    """

    if jp.min() < 0. or jp.max() > 1.:
        raise ValueError("Joint probability matrix is not proper: \
                         Some joint probabilities are outside the range (0, 1)")

    for i in range(jp.shape[0]):
        for j in range(jp.shape[0]):
            ul = min(jp[i, i], jp[j, j])
            ll = max(jp[i, j] + jp[j, j] - 1., 0.)
            ll2 = max(jp[i, j] + jp[i, i] - 1., 0.)

            if jp[i, j] > ul or jp[i, j] < ll or jp[i, j] < ll2:
                raise ValueError(f"Elements ({i, j}) of joint probability are not admissible.")

    return True


def create_joint_prob_corr_table(to_dict=True):
    """
    This function generates the table of joint probabilities for
    each configuration of marginal probabilities and correlation values.
    :param to_dict:
    :return:
    """

    p = np.arange(0., 1.05, 0.05)  # Range of marginal probability
    corr = np.arange(-1., 1.05, 0.05)  # Range of correlations

    table = np.zeros(shape=(len(corr), len(p), len(p)))

    for i in range(len(corr)):

        sig = np.matrix([[1., corr[i]], [corr[i], 1.]])

        for j in range(len(p)):
            for k in range(j, len(p)):

                if corr[i] == -1:
                    jp = max(0., p[j] + p[k] - 1.)
                elif corr[i] == 0.:
                    jp = p[j] * p[k]
                elif corr[i] == 1.:
                    jp = min(p[j], p[k])
                elif p[j] * p[k] == 0. or p[j] == 1 or p[k] == 1.:
                    jp = p[j] * p[k]
                else:
                    jp = mvnun(np.array([0., 0.]), np.array([np.inf, np.inf]),
                               [norm.ppf(p[j]), norm.ppf(p[k])],
                               sig)[0]

                table[i, j, k] = table[i, k, j] = jp

    if to_dict:
        n_table = {}

        # convert to dictionary:
        for j in range(len(p)):
            for k in range(j, len(p)):
                pj = round(p[j], 2)
                pk = round(p[k], 2)
                n_table[(pj, pk)] = n_table[(pj, pk)] = np.array((corr, table[:, j, k]))

        return n_table

    return table


def corr_to_joint_prob(marg_prob, corr_mat):
    """
    This function converts the marginal probabilities and
    correlation matrix and compute the joint probabilities.
    This follows Equation
    """

    marg_prob = np.array(marg_prob)
    corr_mat = np.array(corr_mat)

    jp_mat = np.zeros_like(corr_mat)  # Joint probability matrix

    marg_var = marg_prob * (1. - marg_prob)

    for i in range(jp_mat.shape[0]):
        for j in range(i, jp_mat.shape[0]):
            jp_mat[i, j] = jp_mat[j, i] = (corr_mat[i, j] * np.sqrt(marg_var[i] * marg_var[j]) +
                                           marg_prob[i] * marg_prob[j])

    return jp_mat


def joint_prob_to_sigma(joint_prob):
    """
    This function converts the joint probabilities to the
    multivariate Gaussian covariance matrix Sigma.
    """

    table = create_joint_prob_corr_table()
    sigma = np.diag(np.ones(joint_prob.shape[0]))

    for i in range(sigma.shape[0]):
        for j in range(i + 1, sigma.shape[0]):
            r, jp = table[tuple(sorted((round(joint_prob[i, i], 2), round(joint_prob[j, j], 2))))]
            f = interpolate.interp1d(jp, r)
            sigma[i, j] = sigma[j, i] = f(joint_prob[i, j])

    return sigma


def mvbin(p, rho=None, joint_prob=None, size=1):
    """
    Generates correlated bernoulli random variables with
    marginals equal to `p` and with correlations specified by `rho` OR
    joint probabilities specified by `jp`.
    """

    if type(p) == np.float or len(p) == 1:
        return np.random.binomial(1, p)

    d = len(p)

    if rho is not None:
        if type(rho) == np.floating or len(rho) == 1:
            rho = rho * np.ones(shape=(d, d))
            np.fill_diagonal(rho, 1.)

        joint_prob = corr_to_joint_prob(p, rho)
    elif joint_prob is None:
        raise Exception("Must provide either valid joint probability or rho matrix.")

    # If the joint probability is admissible, generate the samples:
    if admissible_joint_prob(joint_prob):
        # Sample from a multivariate normal:
        mu = norm.ppf(p)
        sigma = joint_prob_to_sigma(joint_prob)

        print(mu)
        print(sigma)

        sample = multivariate_normal.rvs(mu, sigma, size=size)

        # Thresholding
        sample[sample > 0.] = 1.
        sample[sample <= 0.] = 0.

        return sample
