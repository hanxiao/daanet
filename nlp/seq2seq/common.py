# coding=utf-8

import math
import time

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

INF = 1e30


def initializer(): return tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                         mode='FAN_AVG',
                                                                         uniform=True,
                                                                         dtype=tf.float32)


def rand_uniform_initializer(
        mag): return tf.random_uniform_initializer(-mag, mag, seed=314159)


def truc_norm_initializer(
        std): return tf.truncated_normal_initalizer(stddev=std)


def initializer_relu(): return tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                              mode='FAN_IN',
                                                                              uniform=False,
                                                                              dtype=tf.float32)


regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def get_var(name, shape, dtype=tf.float32,
            initializer_fn=initializer,
            regularizer_fn=regularizer, **kwargs):
    return tf.get_variable(name, shape,
                           initializer=initializer_fn,
                           dtype=dtype,
                           regularizer=regularizer_fn, **kwargs)


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def dense(inputs, hidden_size, use_bias=True, scope=None):
    with tf.variable_scope(scope or "dense"):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden_size]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden_size])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            bias = tf.get_variable(
                "bias", [hidden_size], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, bias)
        res = tf.reshape(res, out_shape)
        return res


def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer=regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer=regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def get_scope_name():
    return tf.get_variable_scope().name.split('/')[0]


def make_var(name, shape, trainable=True):
    return tf.get_variable(name, shape,
                           initializer=initializer(),
                           dtype=tf.float32,
                           trainable=trainable,
                           regularizer=regularizer)


def mblock(scope_name, device_name=None, reuse=None):
    def f2(f):
        def f2_v(self, *args, **kwargs):
            start_t = time.time()
            if device_name:
                with tf.device(device_name), tf.variable_scope(scope_name, reuse=reuse):
                    f(self, *args, **kwargs)
            else:
                with tf.variable_scope(scope_name, reuse=reuse):
                    f(self, *args, **kwargs)
            self.logger.info('%s is build in %.4f secs' %
                             (scope_name, time.time() - start_t))

        return f2_v

    return f2


def get_init_state(args, name, q_type, shape):
    hinit_embed = make_var('hinit_ebd_' + name, shape)
    cinit_embed = make_var('cinit_ebd_' + name, shape)
    h_init = tf.expand_dims(
        tf.nn.embedding_lookup(hinit_embed, q_type), axis=0)
    c_init = tf.expand_dims(
        tf.nn.embedding_lookup(cinit_embed, q_type), axis=0)
    cell_init_state = {
        'lstm': lambda: LSTMStateTuple(c_init, h_init),
        'sru': lambda: h_init,
        'gru': lambda: h_init,
        'rnn': lambda: h_init}[args.cell.replace('bi-', '')]()
    return cell_init_state


def highway(x, size=None, activation=tf.nn.relu,
            num_layers=2, scope="highway", dropout=0.0, reuse=None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name="input_projection", reuse=reuse)
        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid,
                     name="gate_%d" % i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation,
                     name="activation_%d" % i, reuse=reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x


def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                                  filter_shape,
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                                       bias_shape,
                                       regularizer=regularizer,
                                       initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
    """
    negative log likelihood loss
    """
    with tf.name_scope(scope, "log_loss"):
        labels = tf.one_hot(labels, tf.shape(
            probs)[1], axis=1, dtype=tf.float32)
        losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
    return losses


def normalize_distribution(p, eps=1e-9):
    p += eps
    norm = tf.reduce_sum(p, axis=1)
    return tf.cast(p, tf.float32) / tf.reshape(norm, (-1, 1))


def kl_divergence(p, q, eps=1e-9):
    p = normalize_distribution(p, eps)
    q = normalize_distribution(q, eps)
    return tf.reduce_sum(p * tf.log(p / q), axis=1)


def get_kl_loss(start_label, start_probs, bandwidth=1.0):
    a = tf.reshape(tf.range(tf.shape(start_probs)[1]), (1, -1))
    b = tf.reshape(start_label, (-1, 1))
    start_true_probs = tf.exp(-tf.cast(tf.squared_difference(a,
                                                             b), tf.float32) / bandwidth)
    return sym_kl_divergence(start_true_probs, start_probs)


def sym_kl_divergence(p, q, eps=1e-9):
    return (kl_divergence(p, q, eps) + kl_divergence(q, p, eps)) / 2.0


def get_conv_feature(x, out_dim, window_len, upsampling=False):
    a = tf.layers.conv1d(x, out_dim, window_len, strides=max(
        int(math.floor(window_len / 2)), 1))
    if upsampling:
        return upsampling_a2b(a, x, out_dim)
    else:
        return a


def upsampling_a2b(a, b, D_a):
    return tf.squeeze(tf.image.resize_images(tf.expand_dims(a, axis=-1), [tf.shape(b)[1], D_a],
                                             method=ResizeMethod.NEAREST_NEIGHBOR), axis=-1)
