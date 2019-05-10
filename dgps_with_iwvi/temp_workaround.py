### Some temporary sample conditionals. Gpflow can do all of this, but there
# are some performance issues with broadcasting that are soon to be corrected.
# When this happens this file is redundant.

import tensorflow as tf

from gpflow.features import InducingPoints, Kuu, Kuf
from gpflow.kernels import Kernel
from gpflow import settings


def independent_multisample_sample_conditional(Xnew: tf.Tensor, feat: InducingPoints, kern: Kernel, f: tf.Tensor, *,
                                               full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    Multisample, single-output GP conditional.

    NB if full_cov=False is required, this functionality can be achieved by reshaping Xnew to SN x D
    nd using conditional. The purpose of this function is to compute full covariances in batch over S samples.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: M x M
    - Kuf: S x M x N
    - Kff: S x N or S x N x N
    ----------
    :param Xnew: data matrix, size S x N x D.
    :param f: data matrix, M x R
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs. Must be False
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     S x N x R
        - variance: S x N x R, S x R x N x N
    """
    if full_output_cov:
        raise NotImplementedError

    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x M

    S, N, D = tf.shape(Xnew)[0], tf.shape(Xnew)[1], tf.shape(Xnew)[2]
    M = tf.shape(Kmm)[0]

    Kmn_M_SN = Kuf(feat, kern, tf.reshape(Xnew, [S * N, D]))  # M x SN
    Knn = kern.K(Xnew) if full_cov else kern.Kdiag(Xnew)  # S x N or S x N x N

    num_func = tf.shape(f)[1]  # (=R)
    Lm = tf.cholesky(Kmm)  # M x M

    # Compute the projection matrix A
    A_M_SN = tf.matrix_triangular_solve(Lm, Kmn_M_SN, lower=True)
    A = tf.transpose(tf.reshape(A_M_SN, [M, S, N]), [1, 0, 2])  # S x M x N

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)  # S x N x N
        fvar = tf.tile(fvar[:, None, :, :], [1, num_func, 1, 1])  # S x R x N x N
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # S x N
        fvar = tf.tile(fvar[:, None, :], [1, num_func, 1])  # S x R x N

    # another backsubstitution in the unwhitened case
    if not white:
        A_M_SN = tf.matrix_triangular_solve(tf.transpose(Lm), A_M_SN, lower=False)
        A = tf.transpose(tf.reshape(A_M_SN, [M, S, N]), [1, 0, 2])  # S x M x N

    # construct the conditional mean
    fmean = tf.matmul(A, tf.tile(f[None, :, :], [S, 1, 1]), transpose_a=True)  # S x N x R
    # fmean = tf.einsum('snm,nr->smr', A, f)  # S x N x R

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A[:, None, :, :] * tf.transpose(q_sqrt)[None, :, :, None]  # S x R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            # L = tf.tile(tf.matrix_band_part(q_sqrt, -1, 0)[None, :, :, :], [S, 1, 1, 1])  # S x R x M x M
            # A_tiled = tf.tile(tf.expand_dims(A, 1), tf.stack([1, num_func, 1, 1]))  # S x R x M x N
            # LTA = tf.matmul(L, A_tiled, transpose_a=True)  # S x R x M x N
            LTA = tf.einsum('rMm,sMn->srmn', tf.matrix_band_part(q_sqrt, -1, 0), A)
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # S x R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 2)  # S x R x N


    if not full_cov:
        z = tf.random_normal(tf.shape(fmean), dtype=settings.float_type)
        fvar = tf.matrix_transpose(fvar)  # S x N x R
        sample = fmean + z * fvar**0.5
    else:
        fmean_SRN1 = tf.transpose(fmean, [0, 2, 1])[:, :, :, None]
        z = tf.random_normal(tf.shape(fmean_SRN1), dtype=settings.float_type)
        sample_SRN1 = fmean + tf.matmul(tf.cholesky(fvar), z)
        sample = tf.transpose(sample_SRN1[:, :, :, 0], [0, 2, 1])

    return sample, fmean, fvar  # fmean is S x N x R, fvar is S x R x N x N or S x N x R


from gpflow.multioutput import Mok, MixedKernelSharedMof
from gpflow import params_as_tensors_for, params_as_tensors
from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import sample_conditional
from gpflow.mean_functions import Linear, Zero

class SharedMixedMok(Mok):
    """
    Linear mixing of the latent GPs, which all share the same kernel to form the output
    """
    def __init__(self, kernel, W, name=None):
        Parameterized.__init__(self, name=name)

        self.kernel = kernel
        self.W = Parameter(W)  # P x L


def multisample_sample_conditional(Xnew: tf.Tensor, feat: InducingPoints, kern: Kernel, f: tf.Tensor, *,
                                   full_cov=False,
                                   full_output_cov=False,
                                   q_sqrt=None,
                                   white=False):
    if isinstance(kern, SharedMixedMok) and isinstance(feat, MixedKernelSharedMof):
        if Xnew.get_shape().ndims == 3:
            sample, gmean, gvar = independent_multisample_sample_conditional(Xnew, feat.feat, kern.kernel, f,
                                                                             white=white,
                                                                             q_sqrt=q_sqrt,
                                                                             full_output_cov=False,
                                                                             full_cov=False)  # N x L, N x L

            o = tf.ones(([tf.shape(Xnew)[0], 1, 1]), dtype=settings.float_type)

        else:
            sample, gmean, gvar = sample_conditional(Xnew, feat.feat, kern.kernel, f,
                                                     white=white,
                                                     q_sqrt=q_sqrt,
                                                     full_output_cov=False,
                                                     full_cov=False)  # N x L, N x L

            o = 1.

        with params_as_tensors_for(kern):
            f_sample = tf.matmul(sample, o * kern.W, transpose_b=True)
            f_mu = tf.matmul(gmean, o * kern.W, transpose_b=True)
            f_var = tf.matmul(gvar, o * kern.W ** 2, transpose_b=True)

        return f_sample, f_mu, f_var
    else:
        assert not isinstance(kern, Mok)
        if Xnew.get_shape().ndims == 3:
            return independent_multisample_sample_conditional(Xnew, feat, kern, f,
                                                              full_cov=full_cov,
                                                              full_output_cov=full_output_cov,
                                                              q_sqrt=q_sqrt,
                                                              white=white)
        else:
            return sample_conditional(Xnew, feat, kern, f,
                                      full_cov=full_cov,
                                      full_output_cov=full_output_cov,
                                      q_sqrt=q_sqrt,
                                      white=white)


from gpflow.logdensities import multivariate_normal
from gpflow.kullback_leiblers import gauss_kl as gauss_kl_gpflow

def gauss_kl(q_mu, q_sqrt, K=None):
    """
    Wrapper for gauss_kl from gpflow that returns the negative log prob if q_sqrt is None. This can be  
    for use in HMC: all that is required is to set q_sqrt to None and this function substitues the
    negative log prob instead of the KL (so no need to set q_mu.prior = gpflow.priors.Gaussian(0, 1)). 
    Also, this allows the use of HMC in the unwhitened case. 
    """
    if q_sqrt is None:
        # return negative log prob with q_mu as 'x', with mean 0 and cov K (or I, if None)
        M, D = tf.shape(q_mu)[0], tf.shape(q_mu)[1]
        I = tf.eye(M, dtype=settings.float_type)

        if K is None:
            L = I
        else:
            L = tf.cholesky(K + I * settings.jitter)

        return -tf.reduce_sum(multivariate_normal(q_mu, tf.zeros_like(q_mu), L))

    else:
        # return kl
        return gauss_kl_gpflow(q_mu, q_sqrt, K=K)

