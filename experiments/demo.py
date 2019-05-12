import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import sys
sys.path.append('../')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


Ns = 300
Xs = np.linspace(-3, 3, Ns).reshape(-1, 1)

f1 = lambda x: np.exp(-(x - 1)**2) + np.exp(-(x + 1)**2) + 0.1 * np.random.randn(*x.shape)**2
f2 = lambda x: np.exp(-(x - 1)**2) + np.random.uniform(low=-0.1, high=0.1, size=x.shape)

np.random.seed(0)

x1 = np.random.uniform(low=-3, high=-0.5, size=(200, 1))
x2 = np.random.uniform(low=0, high=0.5, size=(100, 1))
x3 = np.random.uniform(low=1, high=3, size=(200, 1))

X = np.concatenate([x1, x2, x3], 0)
ind = np.random.choice([True, False], size=X.shape)
Y = np.empty(X.shape)
Y[ind] = f1(X[ind])
Y[np.invert(ind)] = f2(X[np.invert(ind)])

plt.scatter(X, Y, marker='.')
plt.savefig('figs/data.png')

from build_models import build_model

models = []

def add_model(args, name):
    model = build_model(args, X, Y, apply_name=None)
    model.model_name = name
    models.append(model)


class ARGS:
    minibatch_size = None
    lr = 1e-3


class CVAE(ARGS):
    mode = 'CVAE'
    configuration = '200_200'


add_model(CVAE, 'CVAE')


class LG(ARGS):
    mode = 'IWAE'
    M = 50
    likelihood_variance = 0.1
    fix_linear = True
    num_IW_samples = 5
    gamma = 1e-2
    configuration = 'L1_G1'

add_model(LG, 'L1 GP1')


class LGG(LG):
    configuration = 'L1_G1_G1'

add_model(LGG, 'L1 GP1 GP1')


for model in models:
    sess = model.enquire_session()
    model.init_op(sess)

    L = 5
    its = 10000
    fig, axs = plt.subplots(1, L, figsize=(6*L, 6))

    for k, ax in enumerate(axs):
        for it in range(its):
            model.train_op(sess)

        N_samples = 10001
        # Xs_tiled = np.tile(Xs, [N_samples, 1])
        samples = model.predict_y_samples(Xs, N_samples, session=sess)[:, :, 0]

        levels = np.linspace(-0.5, 2.5, 201)
        cs = np.zeros((len(Xs), len(levels)))

        # for i, Ss in enumerate(samples.reshape(N_samples, len(Xs)).T):
        for i, Ss in enumerate(samples.T):
            bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
            kde = KernelDensity(bandwidth=float(bandwidth))

            kde.fit(Ss.reshape(-1, 1))
            for j, level in enumerate(levels):
                cs[i, j] = kde.score(np.array(level).reshape(1, 1))

        # ax.scatter(Xs_tiled, samples, marker='.', alpha=0.2)

        ax.pcolormesh(Xs.flatten(), levels, np.exp(cs.T))#, alpha=0.1)
        ax.scatter(X, Y, marker='x')
        ax.set_title('{} iterations'.format((k + 1) * its))

    # plt.suptitle(model.model_name)
    plt.savefig('figs/{}.png'.format(model.model_name))













