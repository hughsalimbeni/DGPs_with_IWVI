import tensorflow as tf
import numpy as np

import gpflow

from dgps_with_iwvi.layers import RegularizerType


class DGP_VI(gpflow.models.GPModel):
    def __init__(self, X, Y, layers, likelihood,
                 num_samples=1,
                 minibatch_size=None,
                 name=None):
        gpflow.Parameterized.__init__(self, name=name)

        self.likelihood = likelihood

        self.num_data = X.shape[0]
        self.num_samples = num_samples

        if minibatch_size is None:
            self.X = gpflow.params.DataHolder(X)
            self.Y = gpflow.params.DataHolder(Y)
        else:
            self.X = gpflow.params.Minibatch(X, batch_size=minibatch_size, seed=0)
            self.Y = gpflow.params.Minibatch(Y, batch_size=minibatch_size, seed=0)

        self.layers = gpflow.params.ParamList(layers)

    @gpflow.params_as_tensors
    def propagate(self, X, full_cov=False, inference_amorization_inputs=None, is_sampled_local_regularizer=False):

        samples, means, covs, kls, kl_types = [X, ], [], [], [], []

        for layer in self.layers:
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=full_cov,
                                                    inference_amorization_inputs=inference_amorization_inputs,
                                                    is_sampled_local_regularizer=is_sampled_local_regularizer)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)

        return samples[1:], means, covs, kls, kl_types

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        X_tiled = tf.tile(self.X[:, None, :], [1, self.num_samples, 1])  # N, S, Dx
        Y_tiled = tf.tile(self.Y[:, None, :], [1, self.num_samples, 1])  # N, S, Dy

        XY = tf.concat([X_tiled, Y_tiled], -1)  # N, S, Dx+Dy

        # Following Salimbeni 2017, the sampling is independent over N
        # The flag is_sampled_local_regularizer=False means that the KL is returned for the regularizer
        samples, means, covs, kls, kl_types = self.propagate(X_tiled,
                                                             full_cov=False,
                                                             inference_amorization_inputs=XY,
                                                             is_sampled_local_regularizer=False)

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.GLOBAL]

        var_exp = self.likelihood.variational_expectations(means[-1], covs[-1], Y_tiled)  # N, K, Dy

        # Product over the columns of Y
        L_NK = tf.reduce_sum(var_exp, 2)  # N, K, Dy -> N, K

        if len(local_kls) > 0:
            local_kls_NKD = tf.concat(local_kls, -1)  # N, K, sum(W_dims)
            L_NK -= tf.reduce_sum(local_kls_NKD, 2)  # N, K

        scale = tf.cast(self.num_data, gpflow.settings.float_type)\
                / tf.cast(tf.shape(self.X)[0], gpflow.settings.float_type)

        # This line is replaced with tf.reduce_logsumexp(L_NK, 1) - log(S) in the IW case
        logp = tf.reduce_mean(L_NK, 1)

        return tf.reduce_sum(logp) * scale - tf.reduce_sum(global_kls)

    @gpflow.params_as_tensors
    def _build_predict(self, X, full_cov=False):
        _, means, covs, _, _ = self.propagate(X, full_cov=full_cov)
        return means[-1], covs[-1]

    @gpflow.params_as_tensors
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]), (gpflow.settings.int_type, ()))
    def predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]




class DGP_IWVI(DGP_VI):
    @gpflow.params_as_tensors
    def _build_likelihood(self):
        X_tiled = tf.tile(self.X[:, None, :], [1, self.num_samples, 1])  # N, S, Dx
        Y_tiled = tf.tile(self.Y[:, None, :], [1, self.num_samples, 1])  # N, S, Dy

        XY = tf.concat([X_tiled, Y_tiled], -1)  # N, S, Dx+Dy

        # While the sampling independent over N follows just as in Salimbeni 2017, in this
        # case we need to take full cov samples over the multisample dim S.
        # The flag is_sampled_local_regularizer=True means that the log p/q is returned
        # for the regularizer, rather than the KL
        samples, means, covs, kls, kl_types = self.propagate(X_tiled,
                                                             full_cov=True,  # NB the full_cov is over the S dim
                                                             inference_amorization_inputs=XY,
                                                             is_sampled_local_regularizer=True)

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.GLOBAL]

        # This could be made slightly more efficient by making the last layer full_cov=False,
        # but this seems a small price to pay for cleaner code. NB this is only a SxS matrix, not
        # an NxN matrix.
        cov_diag = tf.transpose(tf.matrix_diag_part(covs[-1]), [0, 2, 1])  # N,Dy,K,K -> N,K,Dy
        var_exp = self.likelihood.variational_expectations(means[-1], cov_diag, Y_tiled)  # N, K, Dy

        # Product over the columns of Y
        L_NK = tf.reduce_sum(var_exp, 2)  # N, K, Dy -> N, K

        if len(local_kls) > 0:
            local_kls_NKD = tf.concat(local_kls, -1)  # N, K, sum(W_dims)
            L_NK -= tf.reduce_sum(local_kls_NKD, 2)  # N, K

        scale = tf.cast(self.num_data, gpflow.settings.float_type) \
                / tf.cast(tf.shape(self.X)[0], gpflow.settings.float_type)

        # This is reduce_mean in the VI case.
        logp = tf.reduce_logsumexp(L_NK, 1) - np.log(self.num_samples)

        return tf.reduce_sum(logp) * scale - tf.reduce_sum(global_kls)
