import socket
print(socket.gethostname())

import sys, os
sys.path.append('../')

import numpy as np

# turn off all warnings...
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#################################### args

import argparse

parser = argparse.ArgumentParser()

# model
parser.add_argument("--configuration", default='L1', nargs='?', type=str)
parser.add_argument("--mode", default='IWAE', nargs='?', type=str)
parser.add_argument("--M", default=128, nargs='?', type=int)
parser.add_argument("--likelihood_variance", default=1e-2, nargs='?', type=float)
parser.add_argument("--num_IW_samples", default=5, nargs='?', type=int)

# training
parser.add_argument("--minibatch_size", default=512, nargs='?', type=int)
parser.add_argument("--iterations", default=5000, nargs='?', type=int)
parser.add_argument("--gamma", default=1e-2, nargs='?', type=float)
parser.add_argument("--lr", default=5e-3, nargs='?', type=float)
parser.add_argument("--fix_linear", default=True, nargs='?', type=bool)
parser.add_argument("--num_predict_samples", default=2000, nargs='?', type=int)
parser.add_argument("--predict_batch_size", default=1000, nargs='?', type=int) ## was 10 for experiments

# data
parser.add_argument("--dataset", default='kin8nm', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--seed", default=0, nargs='?', type=int)

parser.add_argument("--results_path", default='havasi_results', nargs='?', type=str)

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

for p in [ARGS.results_path, tensorboard_path_base, checkpoints_path_base, figs_path_base]:
    if not os.path.isdir(p):
        os.mkdir(p)

#################################### data

from bayesian_benchmarks.data import get_regression_data
data = get_regression_data(ARGS.dataset)
data.X_test = data.X_test[:10000]
data.Y_test = data.Y_test[:10000]


#################################### model
from build_models import build_model

model = build_model(ARGS, data.X_train, data.Y_train)





#################################### init

sess = model.enquire_session()
model.init_op(sess)

#################################### monitoring

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
    .with_condition(mon.PeriodicIterationCondition(tensorboard_freq))

monitor_tasks = [print_task, tensorboard_task, checkpoint_task]


#################################### training


with mon.Monitor(monitor_tasks, sess, model.global_step, print_summary=True) as monitor:
    try:
        mon.restore_session(sess, checkpoint_path)
    except ValueError:
        pass

    iterations_to_go = max([ARGS.iterations - sess.run(model.global_step), 0])

    print('Already run {} iterations. Running {} iterations'.format(sess.run(model.global_step), iterations_to_go))

    for it in range(iterations_to_go):
        monitor()
        model.train_op(sess)
    model.anchor(sess)


#################################### evaluation

from sklearn.neighbors import KernelDensity
from scipy.stats import norm, shapiro

res = {}

# Xs_batch = np.array_split(data.X_test, max(1, int(len(data.X_test)/ARGS.predict_batch_size)))


if 'SGHMC' == ARGS.mode:
    spacing = 5
    posterior_samples = model.sghmc_optimizer.collect_samples(sess, ARGS.num_predict_samples, spacing)
    # samples_test = []
    #
    # for x in Xs_batch:
    #     s_batch = np.empty((ARGS.num_predict_samples, len(x), 1))
    #     for i, s in enumerate(posterior_samples):
    #         s_batch[i] = model.predict_y_samples(x, 1, feed_dict=s)[0]
    #     samples_test.append(s_batch)
    # samples_test = np.concatenate(samples_test, 1)

else:
    pass

    # m, v = model.predict_y(data.X_test[:1000])
    # l = norm.logpdf(data.Y_test[:1000], m, v**0.5)
    # print(np.average(l))
    #
    # samples_test = model.predict_y_samples(data.X_test[:10], 1000)
    # print(np.average(samples_test, 0))
    # print(np.std(samples_test, 0))
    # print(m[:10])
    # print(v[:10]**0.5)
    #
    #
    #
    # assert False

    # samples_test = np.concatenate([model.predict_y_samples(x, ARGS.num_predict_samples) for x in Xs_batch], 1)




logp = np.empty(len(data.X_test))
rmse = np.empty(len(data.X_test))
shapiro_W = np.empty(len(data.X_test))

Xs_batch = np.array_split(data.X_test, max(1, int(len(data.X_test)/ARGS.predict_batch_size)))


# for i, (Ss, y) in enumerate(zip(np.transpose(samples_test, [1, 0, 2]), data.Y_test)):
for i, (x, y) in enumerate(zip(data.X_test, data.Y_test)):
    if 'SGHMC' == ARGS.mode:

        samples = np.empty((ARGS.num_predict_samples, 1, 1))
        for j, s in enumerate(posterior_samples):
            samples[j] = model.predict_y_samples(x.reshape(1, -1), 1, feed_dict=s)[0]

    else:
        samples = model.predict_y_samples(x.reshape(1, -1), ARGS.num_predict_samples)

    Ss = samples[:, :, 0]
    bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
    kde = KernelDensity(bandwidth=float(bandwidth))

    l = kde.fit(Ss).score(y.reshape(-1, 1))
    logp[i] = float(l)
    shapiro_W[i] = float(shapiro((Ss - np.average(Ss)) / np.std(Ss))[0])
    rmse[i] = (np.average(Ss) - float(y)) ** 2

res['test_loglik'] = np.average(logp)
res['test_shapiro_W_median'] = np.median(shapiro_W)
res['test_rmse'] = np.average(rmse) ** 0.5

res.update(ARGS.__dict__)
print(res)

#################################### save

from bayesian_benchmarks.database_utils import Database

with Database(results_path) as db:
    db.write('conditional_density_estimation', res)









# import matplotlib
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


# def save_figs():
#     from sklearn.neighbors import KernelDensity
#
#     Xs = data.X_test[:50]
#
#     SS = []
#     num_samples = 10000
#     for x in Xs:
#         samples = model.sample(x.reshape(1, -1), num_samples).reshape(-1, 1)
#
#         # bandwidth = 1.06 * np.std(samples) * num_samples ** (-1. / 5)  # Silverman's (1986) rule of thumb.
#         # kde = KernelDensity(bandwidth=bandwidth)
#         SS.append(samples.flatten())
#
#     SS = np.array(SS)
#     # grid = np.linspace(np.min(SS), np.max(SS), 300).reshape(-1, 1)
#     grid = np.linspace(-3, 3, 300).reshape(-1, 1)
#     # ys = []
#
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=1)
#     cs = pca.fit(data.X_train[:10000, :]).transform(Xs).flatten()
#     cs = (cs - np.min(cs))/(np.max(cs) - np.min(cs))
#
#
#     import matplotlib.cm as cm
#     kde = KernelDensity(bandwidth=1.06 * num_samples ** (-1. / 5))
#
#     for samples, c in zip(SS, cs):
#         normalized_samples = (samples - np.average(samples))/np.std(samples)
#         l = kde.fit(normalized_samples.reshape(-1, 1)).score_samples(grid.reshape(-1, 1))
#         ax.plot(grid.flatten(), np.exp(l).flatten(), alpha=0.2, color=cm.jet(c))
#
#     from scipy.stats import norm
#     y_norm = norm.pdf(grid)
#     ax.plot(grid, y_norm, linestyle=':', color='k')
#
#     import matplotlib as mpl
#     cmap = mpl.cm.jet
#     # norm = mpl.colors.Normalize(vmin=5, vmax=10)
#
#     # ColorbarBase derives from ScalarMappable and puts a colorbar
#     # in a specified axes, so it has everything needed for a
#     # standalone colorbar.  There are many more kwargs, but the
#     # following gives a basic continuous colorbar with ticks
#     # and labels.
#     # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#     #                                 # norm=norm,
#     #                                 orientation='vertical',
#     #                                 )
#
#     # plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(figs_path)
#
#
# save_figs()
#
# # epsilon = 0.01
# # mdecay = 0.05
# # with tf.variable_scope('hmc'):
# #     sghmc_optimizer.generate_update_step(epsilon, mdecay)
# # v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hmc')
# # sess.run(tf.variables_initializer(v))
# #
# # if iterations_to_go > 0:
# #     for it in range(iterations_to_go):
# #         sess.run(op_increment)
# #         monitor()
# #         sghmc_optimizer.sghmc_step(sess)
# #         sghmc_optimizer.train_hypers(sess)
# #
# # num = KDE_ARGS.num_samples  # ARGS.num_predict_samples
# # spacing = 5  # 0
# # posterior_samples = sghmc_optimizer.collect_samples(sess, num, spacing)
# # return posterior_samples
