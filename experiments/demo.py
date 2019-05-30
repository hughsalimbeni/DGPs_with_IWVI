import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import sys, os
sys.path.append('../')

figs_dir = 'figs8'
data_dir = os.path.join(figs_dir, 'data')
if not os.path.isdir(figs_dir):
    os.mkdir(figs_dir)
    os.mkdir(data_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


Ns = 200
Xs = np.linspace(-4, 4, Ns).reshape(-1, 1)

def f(X):
    f0 = lambda x: np.exp(-(x - 1)**2) + np.exp(-(x + 1)**2) - 0.1 * np.exp(np.random.randn(*x.shape)) + 1
    f1 = lambda x: np.exp(-(x - 1)**2) + np.exp(-(x + 1)**2) + 0.1 * np.exp(np.random.randn(*x.shape))
    f2 = lambda x: np.exp(-(x - 1)**2) + np.random.uniform(low=-0.1, high=0.1, size=x.shape)

    ind = np.random.choice([True, False], size=X.shape, p=(0.6, 0.4))
    Y = np.empty(X.shape)
    Y[ind] = f1(X[ind])
    Y[np.invert(ind)] = f2(X[np.invert(ind)])
    # Y[X>2] = f0(X[X>2])
    return Y


np.random.seed(0)

x1 = np.random.uniform(low=-3, high=-0.5, size=(100, 1))
# x2 = np.random.uniform(low=0, high=0.5, size=(, 1))
x3 = np.random.uniform(low=1, high=3, size=(100, 1))

X = np.concatenate([x1, x3], 0)

Y = f(X)



fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X, Y, marker='.', color='C1')
ax.set_ylim(-1, 2)
plt.savefig(os.path.join(data_dir, 'data.png'))
plt.close()


S = 10000

levels = np.linspace(-1, 2, 200)


cs_data = np.zeros((len(Xs), len(levels)))
for i, xs in enumerate(Xs):

    yy = f(np.tile(xs.reshape(1, 1), (S, 1))).flatten()

    bandwidth = 0.01#*1.06 * np.std(yy) * len(yy) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
    kde = KernelDensity(bandwidth=float(bandwidth))
    kde.fit(yy.reshape(-1, 1))

    for j, level in enumerate(levels):
        cs_data[i, j] = kde.score(np.array(level).reshape(1, 1))




for i, (c, xs) in enumerate(zip(cs_data, Xs)):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[1].set_xlim(-0.1, 6)
    axs[0].set_ylim(min(levels), max(levels))

    axs[1].plot(np.exp(c), levels)

    axs[0].set_ylim(min(levels), max(levels))
    axs[0].set_xlim(min(Xs), max(Xs))

    axs[0].pcolormesh(Xs.flatten(), levels, np.exp(cs_data.T), cmap='Blues_r', alpha=1)

    axs[0].scatter(X, Y, marker='.', color='C1')
    axs[0].plot([xs, xs], [-2, 2], color='white', linestyle=':')

    plt.savefig(os.path.join(data_dir, 'data_{:03d}.png'.format(i)))
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

class LG(ARGS):
    mode = 'IWAE'
    M = 50
    likelihood_variance = 0.1
    fix_linear = True
    num_IW_samples = 5
    gamma = 5e-2
    gamma_decay = 0.99
    configuration = 'L1'

class LGG(LG):
    configuration = 'L1_G1'

class LGGG(LG):
    configuration = 'L1_G1_G1'


# add_model(CVAE, 'CVAE')
# add_model(LG, 'LV-GP')
# add_model(LGG,  'LV-GP-GP')
add_model(LGG, 'LV-GP-GP-GP')


def plot_samples(model, path):
    N_samples = 25

    samples = model.predict_y_samples(Xs, N_samples, session=sess)[:, :, 0]
    # objective = np.average([model.compute_log_likelihood() for _ in range(1000)])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    Xs_tiled = np.tile(Xs, [N_samples, 1])

    ax.scatter(Xs_tiled.flatten(), samples.flatten(), marker='.', alpha=0.2, color='C0')

    # ax.set_title('{} iterations. Objective {:.2f}'.format((k + 1) * its, objective))
    ax.set_ylim(-1, 2)
    ax.set_xlim(min(Xs), max(Xs))


    ax.scatter(X, Y, marker='.', color='C1')

    plt.savefig(os.path.join(path, 'samples_{:03d}.png'.format(k)))
    plt.close()


def plot_density(model, path):
    N_samples = 10000

    samples = model.predict_y_samples(Xs, N_samples, session=sess)[:, :, 0]
    # objective = np.average([model.compute_log_likelihood() for _ in range(1000)])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X, Y, marker='.', color='C1')
    levels = np.linspace(-1, 2, 200)
    ax.set_ylim(min(levels), max(levels))
    ax.set_xlim(min(Xs), max(Xs))


    cs = np.zeros((len(Xs), len(levels)))
    for i, Ss in enumerate(samples.T):
        bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
        kde = KernelDensity(bandwidth=float(bandwidth))

        kde.fit(Ss.reshape(-1, 1))
        for j, level in enumerate(levels):
            cs[i, j] = kde.score(np.array(level).reshape(1, 1))
    ax.pcolormesh(Xs.flatten(), levels, np.exp(cs.T), cmap='Blues_r')  # , alpha=0.1)
    ax.scatter(X, Y, marker='.', color='C1')

    plt.savefig(os.path.join(path, 'density_{:03d}.png'.format(k)))
    plt.close()



for model in models:
    sess = model.enquire_session()
    model.init_op(sess)

    path = os.path.join(figs_dir, model.model_name)
    if not os.path.isdir(path):
        os.mkdir(path)

    L = 200
    its = 1000

    for k in range(L):

        for it in range(its):
            model.train_op(sess)

        plot_samples(model, path)
        plot_density(model, path)


    #os.system("convert -quality 100 -delay 10 {}/*.png {}/outvideo.mpeg".format(path, path))
    # s = "ffmpeg -r 10 -i {}/%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p {}/video_{}.mp4"
    # os.system(s.format(path, path, model.model_name))

    S = 10000
    samples = model.predict_y_samples(Xs, S, session=sess)[:, :, 0]
    cs = np.zeros((len(Xs), len(levels)))
    for i, Ss in enumerate(samples.T):
        bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
        kde = KernelDensity(bandwidth=float(bandwidth))

        kde.fit(Ss.reshape(-1, 1))
        for j, level in enumerate(levels):
            cs[i, j] = kde.score(np.array(level).reshape(1, 1))

    for i, (c_data, c, xs) in enumerate(zip(cs_data, cs, Xs)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[1].set_xlim(-0.1, 6)
        axs[0].set_ylim(min(levels), max(levels))

        axs[1].plot(np.exp(c_data), levels, label='data ground truth')
        axs[1].plot(np.exp(c), levels, label='model')

        axs[0].set_ylim(min(levels), max(levels))
        axs[0].set_xlim(min(Xs), max(Xs))

        axs[0].pcolormesh(Xs.flatten(), levels, np.exp(cs.T), cmap='Blues_r', alpha=1)

        axs[0].scatter(X, Y, marker='.', color='C1')
        axs[0].plot([xs, xs], [-2, 2], color='white', linestyle=':')
        plt.legend()

        plt.savefig(os.path.join(path, 'density_with_data_{:03d}.png'.format(i)))
        plt.close()

