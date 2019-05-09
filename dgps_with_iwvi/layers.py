import tensorflow as tf
import numpy as np

import gpflow
import enum


class RegularizerType(enum.Enum):
    LOCAL = 0
    GLOBAL = 1


class GPLayer(gpflow.Parameterized):
    regularizer_type = RegularizerType.GLOBAL
    def __init__(self, kern, Z, num_outputs, mean_function=None, name=None):
        gpflow.Parameterized.__init__(self, name=name)

        self.num_inducing = len(Z)

        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = gpflow.params.Parameter(q_mu)

        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        transform = gpflow.transforms.LowerTriangular(self.num_inducing, num_matrices=num_outputs)
        self.q_sqrt = gpflow.params.Parameter(q_sqrt, transform=transform)

        self.feature = Z if isinstance(Z, gpflow.features.InducingFeature) else gpflow.features.InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function or gpflow.mean_functions.Zero()

        self.num_outputs = num_outputs

    @gpflow.params_as_tensors
    def propagate(self, F, full_cov=False, **kwargs):
        samples, mean, cov = gpflow.conditionals.sample_conditional(F,
                                                                    self.feature,
                                                                    self.kern,
                                                                    self.q_mu,
                                                                    full_cov=full_cov,
                                                                    q_sqrt=self.q_sqrt,
                                                                    white=True)

        kl = gpflow.kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt)

        mf = self.mean_function(F)
        samples += mf
        mean += mf

        return samples, mean, cov, kl


class LatentVariableLayer(gpflow.Parameterized):
    regularizer_type = RegularizerType.LOCAL
    def __init__(self, latent_dim, XY_dim=None, encoder=None, name=None):
        gpflow.Parameterized.__init__(self, name=name)

        self.latent_dim = latent_dim

        # placeholders with default, where the default is the prior
        ones = tf.ones([1, 1], dtype=gpflow.settings.float_type)
        zeros = tf.zeros([1, 1], dtype=gpflow.settings.float_type)
        self.q_mu_placeholder = tf.placeholder_with_default(zeros, [None, None])
        self.q_sqrt_placeholder = tf.placeholder_with_default(ones, [None, None])

        if encoder is None:
            assert XY_dim, 'must pass XY_dim or else an encoder'
            encoder = Encoder(latent_dim, XY_dim, [20, 20])
        self.encoder = encoder

    @gpflow.params_as_tensors
    def propagate(self, F, inference_amorization_inputs=None, is_sampled_local_regularizer=False, **kwargs):
        if inference_amorization_inputs is None:
            """
            If there isn't an X and Y passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            shape = tf.concat([tf.shape(F)[:-1], [self.latent_dim]], 0)
            ones = tf.ones(shape, dtype=gpflow.settings.float_type)
            q_mu = self.q_mu_placeholder * ones  # TODO tf.broadcast_to
            q_sqrt = self.q_sqrt_placeholder * ones  # TODO tf.broadcast_to
        else:
            q_mu, q_sqrt = self.encoder(inference_amorization_inputs)

        # reparameterization trick to take a sample for W
        z = tf.random_normal(tf.shape(q_mu), dtype=gpflow.settings.float_type)
        W = q_mu + z * q_sqrt

        samples = tf.concat([F, W], -1)
        mean = tf.concat([F, q_mu], -1)
        cov = tf.concat([tf.zeros_like(F), q_sqrt ** 2], -1)

        # the prior regularization
        zero, one = [tf.cast(x, dtype=gpflow.settings.float_type) for x in [0, 1]]
        p = tf.contrib.distributions.Normal(zero, one)
        q = tf.contrib.distributions.Normal(q_mu, q_sqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(W) - p.log_prob(W)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tf.contrib.distributions.kl_divergence(q, p)

        return samples, mean, cov, kl


class Encoder(gpflow.Parameterized):
    def __init__(self, latent_dim, input_dim, network_dims, activation_func=None, name=None):
        """
        Encoder that uses GPflow params to encode the features.
        Creates an MLP with input dimensions `input_dim` and produces
        2 * `latent_dim` outputs.
        :param latent_dim: dimension of the latent variable
        :param input_dim: the MLP acts on data of `input_dim` dimensions
        :param network_dims: dimensions of inner MLPs, e.g. [10, 20, 10]
        :param activation_func: TensorFlow operation that can be used
            as non-linearity between the layers (default: tanh).
        """
        gpflow.Parameterized.__init__(self, name=name)
        self.latent_dim = latent_dim
        self.activation_func = activation_func or tf.nn.tanh

        self.layer_dims = [input_dim, *network_dims, latent_dim * 2]

        Ws, bs = [], []

        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            xavier_std = (2. / (input_dim + output_dim)) ** 0.5
            W = np.random.randn(input_dim, output_dim) * xavier_std
            Ws.append(gpflow.Param(W))
            bs.append(gpflow.Param(np.zeros(output_dim)))

        self.Ws, self.bs = gpflow.params.ParamList(Ws), gpflow.params.ParamList(bs)

    @gpflow.params_as_tensors
    def __call__(self, Z):
        o = tf.ones_like(Z)[..., :1, :1]  # for correct broadcasting
        for i, (W, b, dim_in, dim_out) in enumerate(zip(self.Ws, self.bs, self.layer_dims[:-1], self.layer_dims[1:])):
            Z0 = tf.identity(Z)
            Z = tf.matmul(Z, o * W) + o * b

            if i < len(self.bs) - 1:
                Z = self.activation_func(Z)

            if dim_out == dim_in:  # skip connection
                Z += Z0

        means, log_chol_diag = tf.split(Z, 2, axis=-1)
        q_sqrt = tf.nn.softplus(log_chol_diag - 3.)  # bias it towards small vals at first
        q_mu = means
        return q_mu, q_sqrt
