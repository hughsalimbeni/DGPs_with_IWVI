import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import sys, os
sys.path.append('../../../')

# turn off all warnings...
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#################################### args

import argparse

parser = argparse.ArgumentParser()

# model
parser.add_argument("--configuration", default='L1', nargs='?', type=str)
parser.add_argument("--mode", default='IWAE', nargs='?', type=str)  # VI, PRIOR_SAMPLES, GAUSS_HERMITE, IWAE
parser.add_argument("--M", default=128, nargs='?', type=int)
parser.add_argument("--likelihood_variance", default=1e-2, nargs='?', type=float)
parser.add_argument("--num_IW_samples", default=5, nargs='?', type=int)

# training
parser.add_argument("--minibatch_size", default=512, nargs='?', type=int)
parser.add_argument("--iterations", default=5000, nargs='?', type=int)
parser.add_argument("--gamma", default=1e-2, nargs='?', type=float)
parser.add_argument("--lr", default=5e-3, nargs='?', type=float)
parser.add_argument("--fix_linear", default=True, nargs='?', type=bool)
parser.add_argument("--num_predict_samples", default=100, nargs='?', type=int)
parser.add_argument("--predict_batch_size", default=1000, nargs='?', type=int) ## was 10 for experiments

# data
parser.add_argument("--dataset", default='kin8nm', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--seed", default=0, nargs='?', type=int)

parser.add_argument("--results_path", default='/vol/sml/hrs13/dgps_with_iw_fixed_linear_copy_testing', nargs='?', type=str)

ARGS = parser.parse_args()


#################################### paths
if ARGS.split == 0:
    file_name = '{}_{}_{}'.format(ARGS.dataset, ARGS.configuration, ARGS.mode)
else:
    file_name = '{}_{}_{}_{}'.format(ARGS.dataset, ARGS.configuration, ARGS.mode, ARGS.split)

print(file_name)

import os

tensorboard_path_base = os.path.join(ARGS.results_path, 'tensorboard')
checkpoints_path_base = os.path.join(ARGS.results_path, 'checkpoints')
figs_path_base = os.path.join(ARGS.results_path, 'figs')

tensorboard_path = os.path.join(tensorboard_path_base, file_name)
checkpoint_path = os.path.join(checkpoints_path_base, file_name)
figs_path = os.path.join(figs_path_base, file_name+'.png')
results_path = os.path.join(ARGS.results_path, 'results.db')

if not os.path.isdir(ARGS.results_path):
    os.mkdir(ARGS.results_path)
if not os.path.isdir(tensorboard_path_base):
    os.mkdir(tensorboard_path_base)
if not os.path.isdir(checkpoints_path_base):
    os.mkdir(checkpoints_path_base)
if not os.path.isdir(figs_path_base):
    os.mkdir(figs_path_base)


#################################### data

from bayesian_benchmarks.data import get_regression_data
data = get_regression_data(ARGS.dataset)
data.X_test = data.X_test[:10000]
data.Y_test = data.Y_test[:10000]


#################################### model

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.features import InducingPoints
from gpflow.training import NatGradOptimizer, AdamOptimizer
from gpflow.mean_functions import Identity
from gpflow import defer_build

from scipy.cluster.vq import kmeans2
import tensorflow as tf

from gpflux.utils import SharedMixedMok
from gpflux.utils import BroadcastingLinear as Linear
from gpflux.utils import BroadcastingZero as Zero

from gpflow.multioutput.features import MixedKernelSharedMof

from gpflux.layers import GPLayer
from gpflux.layers.latent_variable_layer import LatentVariableConcatLayer
from gpflux.layers.latent_variable_layer import HMCLatentVariableConcatLayer
from gpflux.layers.latent_variable_layer import LatentVariableLayer

from gpflux.models.deep_gp import DeepGP, IWDeepGP
from sghmc import SGHMC

from build_models import CVAE


class KDE_ARGS:
    num_samples = 10000 # was 10000


def make_dgp(ARGS, X, Y):

    if ARGS.mode == 'CVAE':

        layers = []
        for l in ARGS.configuration.split('_'):
            try:
                layers.append(int(l))
            except:
                pass

        with defer_build():
            model = CVAE(data.X_train, data.Y_train, 1, layers, batch_size=ARGS.minibatch_size, name='cvae')

        model.compile()

        global_step = tf.Variable(0, dtype=tf.int32)
        lr = tf.cast(tf.train.exponential_decay(ARGS.lr, global_step, 1000, 0.98, staircase=True), dtype=tf.float64)
        op_adam = AdamOptimizer(lr).make_optimize_tensor(model)
        op_increment = tf.assign_add(global_step, 1)

        ################################################ tensorboard

        import gpflow.training.monitor as mon

        print_freq = 1000
        saving_freq = 500
        tensorboard_freq = 500

        print_task = mon.PrintTimingsTask() \
            .with_name('print') \
            .with_condition(mon.PeriodicIterationCondition(print_freq))
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

        checkpoint_task = mon.CheckpointTask(checkpoint_dir=checkpoint_path, saver=saver) \
            .with_name('checkpoint') \
            .with_condition(mon.PeriodicIterationCondition(saving_freq)) \
            .with_exit_condition(True)

        writer = mon.LogdirWriter(tensorboard_path)
        tensorboard_task = mon.ModelToTensorBoardTask(writer, model) \
            .with_name('tensorboard') \
            .with_condition(mon.PeriodicIterationCondition(tensorboard_freq)) \
            # .with_exit_condition(True)

        monitor_tasks = [print_task, tensorboard_task, checkpoint_task]
        ################################################ training loop
        def optimize():
            sess = model.enquire_session()

            sess.run(tf.variables_initializer([global_step]))

            with mon.Monitor(monitor_tasks, sess, global_step, print_summary=True) as monitor:
                try:
                    mon.restore_session(sess, checkpoint_path)
                except ValueError:
                    pass

                iterations_to_go = max([ARGS.iterations - sess.run(global_step), 0])
                print(
                    'Already run {} iterations. Running {} iterations'.format(sess.run(global_step), iterations_to_go))
                if iterations_to_go > 0:
                    for it in range(iterations_to_go):
                        sess.run(op_increment)
                        monitor()
                        sess.run(op_adam)

        model.optimize = optimize
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
                            # kern = SeparateMixedMok([make_kern() for _ in range(num_gps)], W=PP)
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
                        layers.append(LatentVariableConcatLayer(d, XY_dim=DX + 1))

            kern = RBF(D_in, lengthscales=float(D_in)**0.5, variance=1., ARD=True)
            # if D_in > DX:
            #     ZZ = np.concatenate([Z, np.random.randn(ARGS.M, D_in - DX)], 1)
            ZZ = np.random.randn(ARGS.M, D_in)
            ZZ[:, :min(D_in, DX)] = Z[:, :min(D_in, DX)]
            layers.append(GPLayer(kern, InducingPoints(ZZ), DY, mean_function=Zero()))


            #################################### model

            if ARGS.mode == 'VI':
                model = DeepGP(X, Y, layers,
                               likelihood=lik,
                               batch_size=ARGS.minibatch_size,
                               name='Model')

            elif ARGS.mode == 'HMC':
                for layer in layers:
                    if hasattr(layer, 'q_sqrt'):
                        del layer.q_sqrt
                        layer.q_sqrt = None
                        layer.q_mu.set_trainable(False)

                model = DeepGP(X, Y, layers,
                                  likelihood=lik,
                                  batch_size=ARGS.minibatch_size,
                                  name='Model')



            # elif ARGS.mode == 'HMC_full':
            #     layers_hmc = []
            #     for layer in layers:
            #         if hasattr(layer, 'q_sqrt'):
            #             layer.q_sqrt = None
            #             layer.q_mu.set_trainable(False)
            #
            #             layers_hmc.append(layer)
            #
            #         if isinstance(layer, LatentVariableLayer):
            #             layers_hmc.append(HMCLatentVariableConcatLayer(X.shape[0], layer.latent_dim))
            #
            #     model = DeepGP(X, Y, layers_hmc,
            #                    likelihood=lik,
            #                    batch_size=None,
            #                    name='Model')

            elif ARGS.mode == 'IWAE':
                model = IWDeepGP(X, Y, layers,
                                 likelihood=lik,
                                 batch_size=ARGS.minibatch_size,
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

            if hasattr(model, 'apply_gradients'):
                adam_optimizer = tf.train.AdamOptimizer(lr)
                op_adam = model.apply_gradients(adam_optimizer)

            else:
                op_adam = AdamOptimizer(lr).make_optimize_tensor(model)

        else:
            hyper_train_op = AdamOptimizer(ARGS.lr).make_optimize_tensor(model)
            hmc_vars = []
            for layer in layers:
                if hasattr(layer, 'q_mu'):
                    hmc_vars.append(layer.q_mu.unconstrained_tensor)

            sghmc_optimizer = SGHMC(model, hmc_vars, hyper_train_op, 100)




        ################################################ tensorboard

        import gpflow.training.monitor as mon

        print_freq = 1000
        saving_freq = 500
        tensorboard_freq = 500
        # full_lml_freq = 1000

        print_task = mon.PrintTimingsTask() \
            .with_name('print') \
            .with_condition(mon.PeriodicIterationCondition(print_freq))

        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        checkpoint_task = mon.CheckpointTask(checkpoint_dir=checkpoint_path, saver=saver) \
            .with_name('checkpoint') \
            .with_condition(mon.PeriodicIterationCondition(saving_freq)) \
            .with_exit_condition(True)

        writer = mon.LogdirWriter(tensorboard_path)
        tensorboard_task = mon.ModelToTensorBoardTask(writer, model) \
            .with_name('tensorboard') \
            .with_condition(mon.PeriodicIterationCondition(tensorboard_freq)) \
            # .with_exit_condition(True)

        monitor_tasks = [print_task, tensorboard_task, checkpoint_task]


        ################################################ training loop
        def optimize():
            sess = model.enquire_session()

            sess.run(tf.variables_initializer([global_step]))

            if hasattr(model, 'apply_gradients'):
                sess.run(tf.variables_initializer(adam_optimizer.variables()))


            with mon.Monitor(monitor_tasks, sess, global_step, print_summary=True) as monitor:
                try:
                    mon.restore_session(sess, checkpoint_path)
                except ValueError:
                    pass

                iterations_to_go = max([ARGS.iterations - sess.run(global_step), 0])
                print('Already run {} iterations. Running {} iterations'.format(sess.run(global_step), iterations_to_go))

                if 'HMC' in ARGS.mode:
                    epsilon = 0.01
                    mdecay = 0.05
                    with tf.variable_scope('hmc'):
                        sghmc_optimizer.generate_update_step(epsilon, mdecay)
                    v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hmc')
                    sess.run(tf.variables_initializer(v))

                    if iterations_to_go > 0:
                        for it in range(iterations_to_go):
                            sess.run(op_increment)
                            monitor()
                            sghmc_optimizer.sghmc_step(sess)
                            sghmc_optimizer.train_hypers(sess)

                    num = KDE_ARGS.num_samples # ARGS.num_predict_samples
                    spacing = 5#0
                    posterior_samples = sghmc_optimizer.collect_samples(sess, num, spacing)
                    return posterior_samples

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