import tensorflow as tf
import numpy as np

from scipy.cluster.vq import kmeans2

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.features import InducingPoints
from gpflow.training import NatGradOptimizer, AdamOptimizer
from gpflow.mean_functions import Identity, Linear
from gpflow import defer_build, params_as_tensors
from gpflow.params import Minibatch, DataHolder, Parameter, ParamList
from gpflow import Param, autoflow

from gpflow.multioutput.features import MixedKernelSharedMof
from gpflow.multioutput.kernels import SharedMixedMok

from gpflow.models import Model
from gpflow import transforms
from gpflow import settings

from dgps_with_iwvi.layers import GPLayer, LatentVariableLayer
from dgps_with_iwvi.models import DGP_VI, DGP_IWVI

# from sghmc import SGHMC


class CVAE(Model):
    def __init__(self, X, Y, latent_dim, layers, batch_size=64, name=None):
        super().__init__(name=name)
        self.X_dim = X.shape[1]
        self.Y_dim = Y.shape[1]  # the conditions
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        if batch_size is not None:
            self.X = Minibatch(X, batch_size=batch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=batch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)
        self.latent_dim = latent_dim

        self.variance = Parameter(.05, transform=transforms.positive)

        self.batch_size = batch_size
        shape = (X.shape[0], latent_dim) if (batch_size is None) else (batch_size, latent_dim)
        self.prior_z = tf.distributions.Normal(loc=tf.zeros(shape, dtype=tf.float32),
                                               scale=tf.cast(1.0, dtype=tf.float32))

        self._build_encoder(layers)
        self._build_decoder(layers)

    def _build_decoder(self, layers):
        # latent_dim + y_dim –256–256-1024–784
        Ws, bs = [], []
        dims = [self.latent_dim + self.X_dim] + layers + [self.Y_dim]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            init_xavier_std = (2.0 / (dim_in + dim_out)) ** 0.5
            Ws.append(Param((np.random.randn(dim_in, dim_out) * init_xavier_std).astype(np.float32)))
            bs.append(Param(np.zeros(dim_out).astype(np.float32)))

        self.Ws_dec, self.bs_dec = ParamList(Ws), ParamList(bs)

    @params_as_tensors
    def _eval_decoder(self, Z, X):
        activation = tf.nn.tanh
        X = tf.concat((Z, tf.cast(X, dtype=tf.float32)), axis=1)
        X = tf.cast(X, dtype=tf.float32)
        for i, (W, b) in enumerate(zip(self.Ws_dec, self.bs_dec)):
            X = tf.matmul(X, W) + b
            if i < len(self.bs) - 1:
                X = activation(X)
        return tf.cast(X, dtype=tf.float32)  # N x Dx

    def _build_encoder(self, layers):
        Ws, bs = [], []
        dims = [self.X_dim + self.Y_dim] + layers + [self.latent_dim * 2]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            init_xavier_std = (2.0 / (dim_in + dim_out)) ** 0.5
            Ws.append(Param((np.random.randn(dim_in, dim_out) * init_xavier_std).astype(np.float32)))
            bs.append(Param(np.zeros(dim_out).astype(np.float32)))

        self.Ws, self.bs = ParamList(Ws), ParamList(bs)

    @params_as_tensors
    def _eval_encoder(self, X, Y):
        activation = tf.nn.tanh
        X = tf.concat((X, Y), axis=1)
        X = tf.cast(X, dtype=tf.float32)
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            X = tf.matmul(X, W) + b
            if i < len(self.bs) - 1:
                X = activation(X)

        means, log_chol_diag = tf.split(X, 2, axis=1)
        log_chol_diag = tf.clip_by_value(log_chol_diag, np.log(1e-12), np.log(1e10))

        return tf.cast(means, dtype=tf.float32), tf.cast(log_chol_diag / 2.0, dtype=tf.float32)

    def sample(self, means, log_chol_diag):
        eps = tf.random_normal(tf.shape(means), dtype=tf.float32)
        Z_samples = means + eps * tf.exp(log_chol_diag)
        return Z_samples

    @params_as_tensors
    def _build_likelihood(self):
        means, log_chol_diag = self._eval_encoder(self.X, self.Y)
        Z_samples = self.sample(means, log_chol_diag)
        X_hat = self._eval_decoder(Z_samples, self.X)

        var = tf.cast(self.variance, dtype=tf.float32)

        p_x_hat = tf.distributions.Normal(loc=X_hat, scale=tf.sqrt(var))
        q_z = tf.distributions.Normal(loc=means, scale=tf.exp(log_chol_diag))

        log_prob = tf.reduce_sum(p_x_hat.log_prob(self.Y))  # scalar
        kl = tf.reduce_sum(q_z.kl_divergence(self.prior_z))  # scalar
        elbo = log_prob - kl

        return tf.cast(elbo, settings.float_type)

    @params_as_tensors
    @autoflow((tf.float32, [None, None]))
    def predict_y(self, Xs):
        N = tf.shape(Xs)[0]
        Zs = tf.random_normal((N, self.latent_dim))  # N M x L
        Ys_recon = self._eval_decoder(Zs, Xs)  # N M x Dx
        scale = tf.cast(tf.sqrt(self.variance), tf.float32)
        return Ys_recon, tf.ones_like(Ys_recon) * scale

    @params_as_tensors
    @autoflow((tf.float32, [None, None]), (tf.int32, ()))
    def predict_y_samples(self, Xs, S):
        N = tf.shape(Xs)[0]
        Xs_tiled = tf.tile(Xs, [S, 1])
        Zs = tf.random_normal((S * N, self.latent_dim))  # NS, L
        z = tf.random_normal((S * N, self.Y_dim))
        Ys_recon = self._eval_decoder(Zs, Xs_tiled)  # SN, Y_dim
        scale = tf.cast(tf.sqrt(self.variance), tf.float32)
        samples = Ys_recon + z * tf.ones_like(Ys_recon) * scale
        return tf.reshape(samples, (S, N, self.Y_dim))



def build_model(ARGS, X, Y):

    if ARGS.mode == 'CVAE':

        layers = []
        for l in ARGS.configuration.split('_'):
            try:
                layers.append(int(l))
            except:
                pass

        with defer_build():
            model = CVAE(X, Y, 1, layers, batch_size=ARGS.minibatch_size, name='cvae')

        model.compile()

        global_step = tf.Variable(0, dtype=tf.int32)
        op_increment = tf.assign_add(global_step, 1)

        lr = tf.cast(tf.train.exponential_decay(ARGS.lr, global_step, 1000, 0.98, staircase=True), dtype=tf.float64)
        op_adam = AdamOptimizer(lr).make_optimize_tensor(model)

        model.train_op = lambda s: s.run([op_adam, op_increment])
        model.init_op = lambda s: s.run(tf.variables_initializer([global_step]))
        model.global_step = global_step

        return model

    else:
        N, D = X.shape

        # first layer inducing points
        if N > ARGS.M:
            Z = kmeans2(X, ARGS.M, minit='points')[0]
        else:
            M_pad = ARGS.M - N
            Z = np.concatenate([X.copy(), np.random.randn(M_pad, D)], 0)

        #################################### layers
        P = np.linalg.svd(X, full_matrices=False)[2]
        # PX = P.copy()

        layers = []
        # quad_layers = []

        DX = D
        DY = 1

        D_in = D
        D_out = D
        with defer_build():
            lik = Gaussian()
            lik.variance = ARGS.likelihood_variance

            if len(ARGS.configuration) > 0:
                for c, d in ARGS.configuration.split('_'):
                    if c == 'G':
                        if d == 'X':
                            kern = RBF(D_in, lengthscales=float(D_in) ** 0.5, variance=1., ARD=True)
                            l = GPLayer(kern, InducingPoints(Z), D_in, mean_function=Identity())
                            layers.append(l)
                        else:
                            num_gps = int(d)
                            A = np.zeros((D_in, D_out))
                            D_min = min(D_in, D_out)
                            A[:D_min, :D_min] = np.eye(D_min)
                            mf = Linear(A=A)
                            mf.b.set_trainable(False)

                            def make_kern():
                                k = RBF(D_in, lengthscales=float(D_in) ** 0.5, variance=1., ARD=True)
                                k.variance.set_trainable(False)
                                return k

                            PP = np.zeros((D_out, num_gps))
                            PP[:, :min(num_gps, DX)] = P[:, :min(num_gps, DX)]
                            kern = SharedMixedMok(make_kern(), W=PP)
                            ZZ = np.random.randn(ARGS.M, D_in)
                            ZZ[:, :min(D_in, DX)] = Z[:, :min(D_in, DX)]
                            inducing = MixedKernelSharedMof(InducingPoints(ZZ))
                            l = GPLayer(kern, inducing, num_gps, mean_function=mf)
                            if ARGS.fix_linear is True:
                                kern.W.set_trainable(False)
                                mf.set_trainable(False)

                            layers.append(l)

                            D_in = D_out

                    elif c == 'L':
                        d = int(d)
                        D_in += d
                        layers.append(LatentVariableLayer(d, XY_dim=DX+1))

            kern = RBF(D_in, lengthscales=float(D_in)**0.5, variance=1., ARD=True)
            ZZ = np.random.randn(ARGS.M, D_in)
            ZZ[:, :min(D_in, DX)] = Z[:, :min(D_in, DX)]
            layers.append(GPLayer(kern, InducingPoints(ZZ), DY))


            #################################### model

            if ARGS.mode == 'VI':
                model = DGP_VI(X, Y, layers, lik,
                               minibatch_size=ARGS.minibatch_size,
                               name='Model')

            elif ARGS.mode == 'HMC':
                for layer in layers:
                    if hasattr(layer, 'q_sqrt'):
                        del layer.q_sqrt
                        layer.q_sqrt = None
                        layer.q_mu.set_trainable(False)

                model = DGP_VI(X, Y, layers, lik,
                               minibatch_size=ARGS.minibatch_size,
                               name='Model')


            elif ARGS.mode == 'IWAE':
                model = DGP_IWVI(X, Y, layers, lik,
                                 minbatch_size=ARGS.minibatch_size,
                                 num_samples=ARGS.num_IW_samples,
                                 name='Model')
        model.compile()



        global_step = tf.Variable(0, dtype=tf.int32)
        op_increment = tf.assign_add(global_step, 1)

        if not ('HMC' in ARGS.mode):
            for layer in model.layers[:-1]:
                if isinstance(layer, GPLayer):
                    layer.q_sqrt = layer.q_sqrt.read_value() * 1e-5


            #################################### optimization

            var_list = [[model.layers[-1].q_mu, model.layers[-1].q_sqrt]]

            model.layers[-1].q_mu.set_trainable(False)
            model.layers[-1].q_sqrt.set_trainable(False)

            gamma = tf.cast(tf.train.exponential_decay(ARGS.gamma, global_step, 1000, 0.98, staircase=True),
                            dtype=tf.float64)
            lr = tf.cast(tf.train.exponential_decay(ARGS.lr, global_step, 1000, 0.98, staircase=True), dtype=tf.float64)

            op_ng = NatGradOptimizer(gamma=gamma).make_optimize_tensor(model, var_list=var_list)

            op_adam = AdamOptimizer(lr).make_optimize_tensor(model)

            model.train_op = lambda s: [s.run(op_ng), s.run(op_adam), s.run(op_increment)]
            model.init_op = lambda s: s.run(tf.variables_initializer([global_step]))
            model.global_step = global_step

        else:

            assert False
            # hyper_train_op = AdamOptimizer(ARGS.lr).make_optimize_tensor(model)
            # hmc_vars = []
            # for layer in layers:
            #     if hasattr(layer, 'q_mu'):
            #         hmc_vars.append(layer.q_mu.unconstrained_tensor)
            #
            # sghmc_optimizer = SGHMC(model, hmc_vars, hyper_train_op, 100)
            #
            # model.train_op = lambda s: [s.run(op_increment),
            #                             sghmc_optimizer.sghmc_step(s),
            #                             sghmc_optimizer.train_hypers(s)]
            # def init_op(s):
            #     epsilon = 0.01
            #     mdecay = 0.05
            #     with tf.variable_scope('hmc'):
            #         sghmc_optimizer.generate_update_step(epsilon, mdecay)
            #     v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hmc')
            #     s.run(tf.variables_initializer(v))
            #     s.run(tf.variables_initializer([global_step]))
            #
            # model.init_op = init_op
            # model.global_step = global_step

        ################################################ training loop

        return model
