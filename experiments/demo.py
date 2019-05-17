import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import sys, os
sys.path.append('../')
if not os.path.isdir('figs'):
    os.mkdir('figs')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


Ns = 300
Xs = np.linspace(-3, 3, Ns).reshape(-1, 1)

f1 = lambda x: np.exp(-(x - 1)**2) + np.exp(-(x + 1)**2) + 0.1 * np.random.randn(*x.shape)**2
f2 = lambda x: np.exp(-(x - 1)**2) + np.random.uniform(low=-0.1, high=0.1, size=x.shape)

np.random.seed(0)

x1 = np.random.uniform(low=-3, high=-0.5, size=(100, 1))
# x2 = np.random.uniform(low=0, high=0.5, size=(, 1))
x3 = np.random.uniform(low=1, high=3, size=(100, 1))

X = np.concatenate([x1, x3], 0)
ind = np.random.choice([True, False], size=X.shape, p=(0.6, 0.4))
Y = np.empty(X.shape)
Y[ind] = f1(X[ind])
Y[np.invert(ind)] = f2(X[np.invert(ind)])

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X, Y, marker='x')
ax.set_ylim(-1, 2)
plt.savefig('figs/data.png')
plt.close()

from build_models import build_model

models = []

def add_model(args, name):
    model = build_model(args, X, Y, apply_name=None)
    model.model_name = name
    models.append(model)


class ARGS:
    minibatch_size = None
    lr = 5e-3
    lr_decay = 0.99

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
    gamma = 5e-2
    gamma_decay = 0.99
    configuration = 'L1'

add_model(LG, 'LV-GP')


class LGG(LG):
    configuration = 'L1_G1'

add_model(LGG, 'LV-GP-GP')


class LGGG(LG):
    configuration = 'L1_G1_G1'

add_model(LGG, 'LV-GP-GP-GP')


for model in models:
    sess = model.enquire_session()
    model.init_op(sess)

    L = 50
    its = 5000
    # fig, axs = plt.subplots(1, L, figsize=(6*L, 6))

    # for k, ax in enumerate(axs):
    for k in range(L):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))


        for it in range(its):
            model.train_op(sess)

        N_samples = 25

        samples = model.predict_y_samples(Xs, N_samples, session=sess)[:, :, 0]
        objective = np.average([model.compute_log_likelihood() for _ in range(10000)])

        # levels = np.linspace(-0.5, 2.5, 201)
        # cs = np.zeros((len(Xs), len(levels)))
        # for i, Ss in enumerate(samples.reshape(N_samples, len(Xs)).T):
        # for i, Ss in enumerate(samples.T):
        #     bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
        #     kde = KernelDensity(bandwidth=float(bandwidth))
        #
        #     kde.fit(Ss.reshape(-1, 1))
        #     for j, level in enumerate(levels):
        #         cs[i, j] = kde.score(np.array(level).reshape(1, 1))
        # ax.pcolormesh(Xs.flatten(), levels, np.exp(cs.T))#, alpha=0.1)

        Xs_tiled = np.tile(Xs, [N_samples, 1])

        ax.scatter(Xs_tiled.flatten(), samples.flatten(), marker='.', alpha=0.2, color='C1')

        ax.set_title('{} iterations. Objective {:.2f}'.format((k + 1) * its, objective))
        ax.set_ylim(-1, 2)

        ax.scatter(X, Y, marker='x', color='C0')

    # plt.suptitle(model.model_name)
        path = os.path.join('figs', model.model_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, '{:03d}.png'.format(k)))
        plt.close()


    os.system("convert -quality 100 -delay 20 {}/*.png {}/outvideo.mpeg".format(path, path))












