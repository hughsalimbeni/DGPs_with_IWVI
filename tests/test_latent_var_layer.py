import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import tensorflow as tf
import numpy as np

import gpflow

from dgps_with_iwvi.layers import LatentVariableLayer, GPLayer
from dgps_with_iwvi.models import DGP_VI, DGP_IWVI


class DirectlyParameterizedEncoder(gpflow.Parameterized):
    """
    No amortization is used; each datapoint element has an
    associated mean and variance of its latent variable.

    IMPORTANT: Not compatible with minibatches
    """

    def __init__(self, latent_dim, num_data, num_samples=None, mean=None, std=None, name=None):
        gpflow.Parameterized.__init__(self, name=name)
        self.latent_dim = latent_dim
        self.num_data = num_data

        if mean is None:
            mean = np.random.randn(num_data, latent_dim)

        if mean.shape != (num_data, latent_dim):
            raise ValueError("mean must have shape (num_data={}, latent_dim={})"
                             .format(num_data, latent_dim))

        if std is None:
            std = np.ones((num_data, latent_dim)) * 1e-4

        self.mean = gpflow.Param(mean)
        self.std = gpflow.Param(std, transform=gpflow.transforms.positive)
        self.num_samples = num_samples

    @gpflow.params_as_tensors
    def __call__(self, Z):
        m = tf.tile(self.mean[:, None, :], [1, self.num_samples, 1])
        s = tf.tile(self.std[:, None, :], [1, self.num_samples, 1])
        return m, s


class Data:
    N = 10
    Ns = 100
    Dx = 1
    Dy = 2
    M = 25

    np.random.seed(0)

    Xs = np.random.randn(Ns, Dx)

    X_mean = np.random.randn(N, Dx)
    X_var = np.random.uniform(low=1e-4, high=1e-1, size=(N, Dx))

    Z = np.random.randn(M, Dx)

    Y = np.concatenate([np.sin(10*X_mean), np.cos(10*X_mean)], 1)


class Fixtures:
    kern = gpflow.kernels.RBF(1, lengthscales=0.1)
    lik = gpflow.likelihoods.Gaussian()  # var = 1


custom_config = gpflow.settings.get_settings()
custom_config.numerics.jitter_level = 1e-18
with gpflow.settings.temp_settings(custom_config):

    class ReferenceBGPLVM:
        model = gpflow.models.BayesianGPLVM(Data.X_mean,
                                            Data.X_var,
                                            Data.Y,
                                            Fixtures.kern,
                                            Data.M,
                                            Z=Data.Z)

        # get inducing point distribution
        mu, cov = model.predict_f_full_cov(Data.Z)

        # unwhiten
        std = np.linalg.cholesky(np.transpose(cov, [2, 0, 1]))
        K = Fixtures.kern.compute_K_symm(Data.Z)
        L = np.linalg.cholesky(K)
        L_inv = np.linalg.inv(L)
        cov_white = np.einsum('ab,bcp,dc->pad', L_inv, cov, L_inv)

        q_mu = np.linalg.solve(L, mu)
        q_sqrt = np.linalg.cholesky(cov_white + 1e-12 * np.eye(Data.M)[None, :, :])

        # predictions
        pred_m, pred_v = model.predict_f(Data.Xs)
        pred_m_full_cov, pred_v_full_cov = model.predict_f_full_cov(Data.Xs)

        # bound
        L = model.compute_log_likelihood()


def test_bound_vs_gplvm():
    custom_config = gpflow.settings.get_settings()
    custom_config.numerics.jitter_level = 1e-18
    with gpflow.settings.temp_settings(custom_config):
        encoder = DirectlyParameterizedEncoder(Data.Dx, Data.N,
                                               mean=Data.X_mean,
                                               std=Data.X_var**0.5,
                                               num_samples=1)

        layers = [LatentVariableLayer(Data.Dx, encoder=encoder),
                  GPLayer(Fixtures.kern, Data.Z, Data.Dy)]

        m_dgp_vi = DGP_VI(np.zeros((Data.N, 0)), Data.Y, layers, Fixtures.lik, num_samples=1)
        m_dgp_iw = DGP_IWVI(np.zeros((Data.N, 0)), Data.Y, layers, Fixtures.lik, num_samples=1)

        for model in [m_dgp_vi, m_dgp_iw]:
            model.layers[1].q_mu = ReferenceBGPLVM.q_mu
            model.layers[1].q_sqrt = ReferenceBGPLVM.q_sqrt

            L = [model.compute_log_likelihood() for _ in range(1000)]

            L_mean = np.average(L)
            L_stderr = np.std(L) / len(L) ** 0.5

            # check ground truth is within +-3 std deviation CI
            assert L_mean + 3 * L_stderr > ReferenceBGPLVM.L
            assert L_mean - 3 * L_stderr < ReferenceBGPLVM.L


            m, v = model.predict_f(Data.Xs)
            np.testing.assert_allclose(m, ReferenceBGPLVM.pred_m, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(v, ReferenceBGPLVM.pred_v, atol=1e-6, rtol=1e-6)

            m_full, v_full = model.predict_f_full_cov(Data.Xs)
            v_full = np.transpose(v_full, [1, 2, 0])
            np.testing.assert_allclose(m_full, ReferenceBGPLVM.pred_m_full_cov, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(v_full, ReferenceBGPLVM.pred_v_full_cov, atol=1e-6, rtol=1e-6)


def test_IW_vs_VI():
    K = 10
    encoder = DirectlyParameterizedEncoder(Data.Dx, Data.N,
                                           mean=Data.X_mean,
                                           std=Data.X_var ** 0.5,
                                           num_samples=K)

    layers = [LatentVariableLayer(Data.Dx, encoder=encoder),
              GPLayer(Fixtures.kern, Data.Z, Data.Dy)]

    m_dgp_vi = DGP_VI(np.zeros((Data.N, 0)), Data.Y, layers, Fixtures.lik, num_samples=K)
    m_dgp_iw = DGP_IWVI(np.zeros((Data.N, 0)), Data.Y, layers, Fixtures.lik, num_samples=K)

    for model in [m_dgp_vi, m_dgp_iw]:
        model.layers[1].q_mu = ReferenceBGPLVM.q_mu
        model.layers[1].q_sqrt = ReferenceBGPLVM.q_sqrt

    L_vi = [m_dgp_vi.compute_log_likelihood() for _ in range(1000)]
    L_iw = [m_dgp_iw.compute_log_likelihood() for _ in range(1000)]

    L_vi_mean = np.average(L_vi)
    L_iw_mean = np.average(L_iw)

    # for K > 1 the IW estimate should be greater than the VI estimator
    assert L_vi_mean < L_iw_mean


def test_IW_var_vs_VI_single_sample():
    K = 1
    encoder = DirectlyParameterizedEncoder(Data.Dx, Data.N,
                                           mean=Data.X_mean,
                                           std=Data.X_var ** 0.5,
                                           num_samples=K)

    layers = [LatentVariableLayer(Data.Dx, encoder=encoder),
              GPLayer(Fixtures.kern, Data.Z, Data.Dy)]

    m_dgp_vi = DGP_VI(np.zeros((Data.N, 0)), Data.Y, layers, Fixtures.lik, num_samples=K)
    m_dgp_iw = DGP_IWVI(np.zeros((Data.N, 0)), Data.Y, layers, Fixtures.lik, num_samples=K)

    for model in [m_dgp_vi, m_dgp_iw]:
        model.layers[1].q_mu = ReferenceBGPLVM.q_mu
        model.layers[1].q_sqrt = ReferenceBGPLVM.q_sqrt

    L_vi = [m_dgp_vi.compute_log_likelihood() for _ in range(1000)]
    L_iw = [m_dgp_iw.compute_log_likelihood() for _ in range(1000)]

    L_vi_std = np.std(L_vi)
    L_iw_std = np.std(L_iw)

    # in the 1 sample case the variance of the VI estimator should be strictly less than the IW
    assert L_vi_std < L_iw_std

