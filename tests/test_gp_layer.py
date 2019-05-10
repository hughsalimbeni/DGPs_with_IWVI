import numpy as np
import pytest

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import gpflow
import sys
sys.path.append('../')

from dgps_with_iwvi.layers import GPLayer
from dgps_with_iwvi.models import DGP_VI


def test_gp_layer():
    N = 10001
    M = 100
    Dy = 1

    np.random.seed(0)

    X = np.linspace(0, 1, N).reshape(-1, 1)
    Z = np.linspace(0, 1, M).reshape(-1, 1)
    Xs = np.linspace(0, 1, N-1).reshape(-1, 1)

    Y = np.concatenate([np.sin(10*X), np.cos(10*X)], 1)[:, 0:1]

    kern = gpflow.kernels.Matern52(1, lengthscales=0.1)
    mean_function = gpflow.mean_functions.Linear(A=np.random.randn(1, Dy))
    lik = gpflow.likelihoods.Gaussian(variance=1e-1)

    m_vgp = gpflow.models.SVGP(X, Y, kern, lik, Z=Z,
                               mean_function=mean_function)

    q_mu = np.random.randn(M, Dy)
    q_sqrt = np.random.randn(Dy, M, M)

    m_vgp.q_mu = q_mu
    m_vgp.q_sqrt = q_sqrt

    m1, v1 = m_vgp.predict_f_full_cov(Xs)
    L1 = m_vgp.compute_log_likelihood()

    m_dgp = DGP_VI(X, Y, [GPLayer(kern, Z, Dy, mean_function)], lik, num_samples=1)

    m_dgp.layers[0].q_mu = q_mu
    m_dgp.layers[0].q_sqrt = q_sqrt

    m2, v2 = m_dgp.predict_f_full_cov(Xs)
    L2 = m_dgp.compute_log_likelihood()

    np.testing.assert_allclose(L1, L2)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(v1, v2)


def test_dgp_zero_inner_layers():
    N = 10
    Dy = 2

    X = np.linspace(0, 1, N).reshape(-1, 1)
    Xs = np.linspace(0, 1, N-1).reshape(-1, 1)

    Y = np.concatenate([np.sin(10*X), np.cos(10*X)], 1)

    kern = gpflow.kernels.Matern52(1, lengthscales=0.1)
    mean_function = gpflow.mean_functions.Linear(A=np.random.randn(1, 2))
    lik = gpflow.likelihoods.Gaussian(variance=1e-1)

    m_vgp = gpflow.models.SVGP(X, Y, kern, lik, Z=X,
                               mean_function=mean_function)

    q_mu = np.random.randn(N, Dy)
    q_sqrt = np.random.randn(Dy, N, N)

    m_vgp.q_mu = q_mu
    m_vgp.q_sqrt = q_sqrt

    m1, v1 = m_vgp.predict_f_full_cov(Xs)

    custom_config = gpflow.settings.get_settings()
    custom_config.numerics.jitter_level = 1e-18
    with gpflow.settings.temp_settings(custom_config):
        m_dgp = DGP_VI(X, Y, [
            GPLayer(gpflow.kernels.RBF(1, variance=1e-6), X, 1, gpflow.mean_functions.Identity()),
            GPLayer(kern, X, Dy, mean_function)], lik)

        m_dgp.layers[-1].q_mu = q_mu
        m_dgp.layers[-1].q_sqrt = q_sqrt

        m_dgp.layers[0].q_sqrt = m_dgp.layers[0].q_sqrt.read_value() * 1e-12

        m2, v2 = m_dgp.predict_f_full_cov(Xs)

        np.testing.assert_allclose(m1, m2, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(v1, v2, atol=1e-5, rtol=1e-5)
