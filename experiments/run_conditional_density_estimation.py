import sys
sys.path.append('../')
import numpy as np


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


if 'SGHMC' == ARGS.mode:
    spacing = 5
    posterior_samples = model.sghmc_optimizer.collect_samples(sess, ARGS.num_predict_samples, spacing)


logp = np.empty(len(data.X_test))
rmse = np.empty(len(data.X_test))
shapiro_W = np.empty(len(data.X_test))

Xs_batch = np.array_split(data.X_test, max(1, int(len(data.X_test)/ARGS.predict_batch_size)))


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
