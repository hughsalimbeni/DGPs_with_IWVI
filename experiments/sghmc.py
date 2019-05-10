# This code is adapted from https://github.com/cambridge-mlg/sghmc_dgp

import numpy as np
import tensorflow as tf

from dgps_with_iwvi.models import DGP_VI


class SGHMC(object):
    def __init__(self, model, vars, hyper_train_op, window_size):
        self.model = model
        self.hyper_train_op = hyper_train_op
        self.vars = vars
        self.window_size = window_size
        self.window = []
        self.sample_op = None
        self.burn_in_op = None

    def generate_update_step(self, epsilon, mdecay):
        nll = -self.model.likelihood_tensor
        self.epsilon = epsilon
        burn_in_updates = []
        sample_updates = []

        grads = tf.gradients(nll, self.vars)

        for theta, grad in zip(self.vars, grads):
            xi = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            g = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            g2 = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            p = tf.Variable(tf.zeros_like(theta), dtype=tf.float64, trainable=False)

            r_t = 1. / (xi + 1.)
            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad ** 2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (tf.sqrt(g2 + 1e-16) + 1e-16)

            burn_in_updates.append((xi, xi_t))
            burn_in_updates.append((g, g_t))
            burn_in_updates.append((g2, g2_t))

            epsilon_scaled = epsilon / tf.sqrt(tf.cast(self.model.num_data, tf.float64))
            noise_scale = 2. * epsilon_scaled ** 2 * mdecay * Minv
            sigma = tf.sqrt(tf.maximum(noise_scale, 1e-16))
            sample_t = tf.random_normal(tf.shape(theta), dtype=tf.float64) * sigma
            p_t = p - epsilon ** 2 * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t

            sample_updates.append((theta, theta_t))
            sample_updates.append((p, p_t))

        self.sample_op = [tf.assign(var, var_t) for var, var_t in sample_updates]
        self.burn_in_op = [tf.assign(var, var_t) for var, var_t in burn_in_updates + sample_updates]

    def collect_samples(self, session, num, spacing):
        posterior_samples = []
        for i in range(num):
            for j in range(spacing):
                session.run((self.sample_op))

            values = session.run((self.vars))
            sample = {}
            for var, value in zip(self.vars, values):
                sample[var] = value
            posterior_samples.append(sample)
        return posterior_samples

    def sghmc_step(self, session):
        session.run(self.burn_in_op)
        values = session.run((self.vars))
        sample = {}
        for var, value in zip(self.vars, values):
            sample[var] = value
        self.window.append(sample)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

    def train_hypers(self, session):
        feed_dict = {}
        i = np.random.randint(len(self.window))
        feed_dict.update(self.window[i])
        session.run(self.hyper_train_op, feed_dict=feed_dict)
