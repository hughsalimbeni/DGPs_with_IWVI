import tensorflow as tf
from tensorflow.contrib.distributions import MvnDiag

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

from sghmc import SGHMC


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
        self.prior_z = MvnDiag(loc=tf.zeros(shape, dtype=tf.float32), scale_identity_multiplier=tf.cast(1.0, dtype=tf.float32))

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

        p_x_hat = MvnDiag(loc=X_hat, scale_identity_multiplier=tf.sqrt(var))
        q_z = MvnDiag(loc=means, scale_diag=tf.exp(log_chol_diag))

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



def make_dgp(ARGS, X, Y):

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
        lr = tf.cast(tf.train.exponential_decay(ARGS.lr, global_step, 1000, 0.98, staircase=True), dtype=tf.float64)
        op_adam = AdamOptimizer(lr).make_optimize_tensor(model)
        op_increment = tf.assign_add(global_step, 1)
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
            hyper_train_op = AdamOptimizer(ARGS.lr).make_optimize_tensor(model)
            hmc_vars = []
            for layer in layers:
                if hasattr(layer, 'q_mu'):
                    hmc_vars.append(layer.q_mu.unconstrained_tensor)

            sghmc_optimizer = SGHMC(model, hmc_vars, hyper_train_op, 100)

            model.train_op = lambda s: [s.run(op_increment),
                                        sghmc_optimizer.sghmc_step(s),
                                        sghmc_optimizer.train_hypers(s)]
            def init_op(s):
                epsilon = 0.01
                mdecay = 0.05
                with tf.variable_scope('hmc'):
                    sghmc_optimizer.generate_update_step(epsilon, mdecay)
                v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hmc')
                s.run(tf.variables_initializer(v))
                s.run(tf.variables_initializer([global_step]))

            model.init_op = init_op
            model.global_step = global_step

        ################################################ training loop
        def optimize():
            sess = model.enquire_session()



                else:
                    if iterations_to_go > 0:
                        for it in range(iterations_to_go):
                            sess.run(op_increment)
                            monitor()
                            sess.run(op_ng)
                            sess.run(op_adam)


        model.optimize = optimize
        return model



import numpy as np

class Model:
    def __init__(self, **kwargs):
        self.model = None

    def fit(self, X, Y):
        self.model = make_dgp(ARGS, X, Y)
        self.model.compile()
        self.samples = self.model.optimize()

    def sample(self, Xs, num_samples):
        if self.samples is None:
            Xs_tiled = np.tile(Xs, [num_samples, 1])
            m, v = self.model.predict_y(Xs_tiled)
            s = m + np.random.randn(*m.shape) * v**0.5
            return s.reshape(num_samples, Xs.shape[0], m.shape[1])
        else:
            s = np.zeros((len(self.samples), Xs.shape[0], 1))
            for i, sample in enumerate(self.samples):
                m, v = self.model.predict_y(Xs, feed_dict=sample)
                s[i] = m + np.random.randn(*m.shape) * v ** 0.5
            return s


#################################### run

# from bayesian_benchmarks.tasks.conditional_density_estimation import run
from bayesian_benchmarks.models.get_model import get_regression_model

from sklearn.neighbors import KernelDensity
from scipy.stats import norm, shapiro



do_saving = True

import time


def run(ARGS, data=None, model=None, is_test=False):

    data = data or get_regression_data(ARGS.dataset, split=ARGS.split)
    model = model or get_regression_model(ARGS.model)(is_test=is_test, seed=ARGS.seed)

    model.fit(data.X_train, data.Y_train)

    res = {}

    if do_saving:
        logp = np.empty(len(data.X_test))
        rmse = np.empty(len(data.X_test))
        shapiro_W = np.empty(len(data.X_test))

        t = time.time()
        Ss = model.sample(data.X_test, KDE_ARGS.num_samples)
        print('predicting all {} in {:.1f}s'.format(len(data.X_test), time.time() - t))
        # for i, (x, y) in enumerate(zip(data.X_test, data.Y_test)):
        for i, (samples, y) in enumerate(zip(np.transpose(Ss, [1, 0, 2]), data.Y_test)):
            # if i % 100 == 0:
            #     print('predicting {} of {} in {:.1f}s'.format(i, len(data.X_test), time.time() - t))
            #     t = time.time()

            # samples = model.sample(x.reshape(1, -1), KDE_ARGS.num_samples).reshape(-1, 1)

            bandwidth = 1.06 * np.std(samples) * KDE_ARGS.num_samples**(-1./5)  # Silverman's (1986) rule of thumb.
            kde = KernelDensity(bandwidth=float(bandwidth))

            l = kde.fit(samples).score(y.reshape(-1, 1))
            logp[i] = float(l)
            shapiro_W[i] = float(shapiro((samples - np.average(samples))/np.std(samples))[0])
            rmse[i] = (np.average(samples) - float(y))**2

        res['test_loglik'] = np.average(logp)
        res['test_shapiro_W_median'] = np.median(shapiro_W)
        res['test_rmse'] = np.average(rmse)**0.5

        # n_max = 10000
        # I = min(n_max, len(data.X_train))
        # logp = np.empty(I)
        # rmse = np.empty(I)
        # shapiro_W = np.empty(I)
        #
        # for i, (x, y) in enumerate(zip(data.X_train[:n_max, :], data.Y_train[:n_max, :])):
        #     samples = model.sample(x.reshape(1, -1), KDE_ARGS.num_samples).reshape(-1, 1)
        #
        #     # samples = samples[np.invert(np.isnan(samples))]
        #
        #     bandwidth = 1.06 * np.std(samples) * KDE_ARGS.num_samples**(-1./5)  # Silverman's (1986) rule of thumb.
        #     kde = KernelDensity(bandwidth=float(bandwidth))
        #
        #     l = kde.fit(samples).score(y.reshape(-1, 1))
        #     logp[i] = float(l)
        #     shapiro_W[i] = float(shapiro((samples - np.average(samples))/np.std(samples))[0])
        #     rmse[i] = (np.average(samples) - float(y))**2
        #
        # res['train_loglik'] = np.average(logp)
        # res['train_shapiro_W_median'] = np.median(shapiro_W)
        # res['train_rmse'] = np.average(rmse)**0.5

        res.update(ARGS.__dict__)

        if not is_test:  # pragma: no cover
            with Database(ARGS.database_path) as db:
                db.write('conditional_density_estimation', res)

    return res

model = Model()
res = run(ARGS, model=model, data=data, is_test=True)
# model.fit(data.X_train, data.Y_train)
print(res)

#################################### save
if do_saving:
    from bayesian_benchmarks.database_utils import Database
    database_path = os.path.join(ARGS.results_path, 'results.db')

    with Database(database_path) as db:
        db.write('conditional_density_estimation', res)

def save_figs():
    from sklearn.neighbors import KernelDensity

    Xs = data.X_test[:50]

    SS = []
    num_samples = 10000
    for x in Xs:
        samples = model.sample(x.reshape(1, -1), num_samples).reshape(-1, 1)

        # bandwidth = 1.06 * np.std(samples) * num_samples ** (-1. / 5)  # Silverman's (1986) rule of thumb.
        # kde = KernelDensity(bandwidth=bandwidth)
        SS.append(samples.flatten())

    SS = np.array(SS)
    # grid = np.linspace(np.min(SS), np.max(SS), 300).reshape(-1, 1)
    grid = np.linspace(-3, 3, 300).reshape(-1, 1)
    # ys = []

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    cs = pca.fit(data.X_train[:10000, :]).transform(Xs).flatten()
    cs = (cs - np.min(cs))/(np.max(cs) - np.min(cs))


    import matplotlib.cm as cm
    kde = KernelDensity(bandwidth=1.06 * num_samples ** (-1. / 5))

    for samples, c in zip(SS, cs):
        normalized_samples = (samples - np.average(samples))/np.std(samples)
        l = kde.fit(normalized_samples.reshape(-1, 1)).score_samples(grid.reshape(-1, 1))
        ax.plot(grid.flatten(), np.exp(l).flatten(), alpha=0.2, color=cm.jet(c))

    from scipy.stats import norm
    y_norm = norm.pdf(grid)
    ax.plot(grid, y_norm, linestyle=':', color='k')

    import matplotlib as mpl
    cmap = mpl.cm.jet
    # norm = mpl.colors.Normalize(vmin=5, vmax=10)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 # norm=norm,
    #                                 orientation='vertical',
    #                                 )

    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(figs_path)


save_figs()