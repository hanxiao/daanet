# coding=utf-8

import tensorflow as tf
import tensorflow.contrib as tc

import gpu_env
from .common import dropout, dense, get_var


def single_rnn_cell(cell_name, num_units, is_train=None, keep_prob=0.75):
    """
    Get a single rnn cell
    """
    cell_name = cell_name.upper()
    if cell_name == "GRU":
        cell = tf.contrib.rnn.GRUCell(num_units)
    elif cell_name == "LSTM":
        cell = tf.contrib.rnn.LSTMCell(num_units)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(num_units)

    # dropout wrapper
    if is_train and keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob)

    return cell


def multi_rnn_cell(cell_name, num_units, is_train=None, keep_prob=1.0, num_layers=3):
    cell_name = cell_name.upper()
    if cell_name == "GRU":
        cells = [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_layers)]
    elif cell_name == "LSTM":
        cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(num_units) for _ in range(num_layers)]
    else:
        cells = [tf.contrib.rnn.BasicRNNCell(num_units) for _ in range(num_layers)]

    cell = tf.contrib.rnn.MultiRNNCell(cells)
    # dropout wrapper
    if is_train and keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob)
    return cell


def get_lstm_init_state(batch_size, num_layers, num_units, direction, scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'lstm_init_state', reuse=reuse):
        num_dir = 2 if direction.startswith('bi') else 1
        c = get_var('lstm_init_c', shape=[num_layers * num_dir, num_units])
        c = tf.tile(tf.expand_dims(c, axis=1), [1, batch_size, 1])
        h = get_var('lstm_init_h', shape=[num_layers * num_dir, num_units])
        h = tf.tile(tf.expand_dims(h, axis=1), [1, batch_size, 1])
        return c, h


def LSTM_encode(seqs, scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'lstm_encode_block', reuse=reuse):
        batch_size = tf.shape(seqs)[0]
        # to T, B, D
        _seqs = tf.transpose(seqs, [1, 0, 2])
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(**kwargs)
        init_state = get_lstm_init_state(batch_size, **kwargs)
        output, state = lstm(_seqs, init_state)
        return tf.transpose(output, [1, 0, 2]), state


def custom_dynamic_rnn(cell, inputs, inputs_len, initial_state=None):
    """
    Implements a dynamic rnn that can store scores in the pointer network,
    the reason why we implements this is that the raw_rnn or dynamic_rnn function in Tensorflow
    seem to require the hidden unit and memory unit has the same dimension, and we cannot
    store the scores directly in the hidden unit.
    Args:
        cell: RNN cell
        inputs: the input sequence to rnn
        inputs_len: valid length
        initial_state: initial_state of the cell
    Returns:
        outputs and state
    """
    batch_size = tf.shape(inputs)[0]
    max_time = tf.shape(inputs)[1]

    inputs_ta = tf.TensorArray(dtype=gpu_env.DTYPE_F, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    emit_ta = tf.TensorArray(dtype=gpu_env.DTYPE_F, dynamic_size=True, size=0)
    t0 = tf.constant(0, dtype=tf.int32)
    if initial_state is not None:
        s0 = initial_state
    else:
        s0 = cell.zero_state(batch_size, dtype=gpu_env.DTYPE_F)
    f0 = tf.zeros([batch_size], dtype=tf.bool)

    def loop_fn(time, prev_s, emit_ta, finished):
        """
        the loop function of rnn
        """
        cur_x = inputs_ta.read(time)
        scores, cur_state = cell(cur_x, prev_s)

        # copy through
        scores = tf.where(finished, tf.zeros_like(scores), scores)

        if isinstance(cell, tc.rnn.LSTMCell):
            cur_c, cur_h = cur_state
            prev_c, prev_h = prev_s
            cur_state = tc.rnn.LSTMStateTuple(tf.where(finished, prev_c, cur_c),
                                              tf.where(finished, prev_h, cur_h))
        else:
            cur_state = tf.where(finished, prev_s, cur_state)

        emit_ta = emit_ta.write(time, scores)
        finished = tf.greater_equal(time + 1, inputs_len)
        return [time + 1, cur_state, emit_ta, finished]

    _, state, emit_ta, _ = tf.while_loop(
        cond=lambda _1, _2, _3, finished: tf.logical_not(
            tf.reduce_all(finished)),
        body=loop_fn,
        loop_vars=(t0, s0, emit_ta, f0),
        parallel_iterations=32,
        swap_memory=False)

    outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
    return outputs, state


def reduce_state(fw_state, bw_state, hidden_size, act_fn=tf.nn.relu, scope=None):
    # concatennation of fw and bw cell
    with tf.variable_scope(scope or "reduce_final_state"):
        _c = tf.concat([fw_state.c, bw_state.c], axis=1)
        _h = tf.concat([fw_state.h, bw_state.h], axis=1)

        c = act_fn(dense(_c, hidden_size, use_bias=True, scope="reduce_c"))
        h = act_fn(dense(_h, hidden_size, use_bias=True, scope="reduce_h"))
        return tc.rnn.LSTMStateTuple(c, h)


class CudaRNN:
    def __init__(self, num_layers, num_units, cell_type):
        self.num_layers = num_layers
        self.num_units = num_units
        if cell_type.endswith('gru'):
            self.grus = [(tf.contrib.cudnn_rnn.CudnnGRU(1, num_units),
                          tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)) for _ in range(num_layers)]
        elif cell_type.endswith('lstm'):
            self.grus = [(tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units),
                          tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units)) for _ in range(num_layers)]
        else:
            raise NotImplementedError

    def __call__(self, inputs, seq_len, keep_prob=1.0,
                 is_train=None, init_states=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        batch_size = tf.shape(inputs)[0]
        if not init_states:
            init_states = []
            for layer in range(self.num_layers):
                init_fw = tf.tile(tf.Variable(
                    tf.zeros([1, 1, self.num_units])), [1, batch_size, 1])
                init_bw = tf.tile(tf.Variable(
                    tf.zeros([1, 1, self.num_units])), [1, batch_size, 1])
                init_states.append((init_fw, init_bw))

        dropout_mask = []
        for layer in range(self.num_layers):
            input_size_ = inputs.get_shape().as_list(
            )[-1] if layer == 0 else 2 * self.num_units
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)

            dropout_mask.append((mask_fw, mask_bw))

        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = init_states[layer]
            mask_fw, mask_bw = dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw,))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw,))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw,))
            self.inits.append((init_fw, init_bw,))
            self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res
