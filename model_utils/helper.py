import json
import logging
import os
import time

import tensorflow as tf

from gpu_env import APP_NAME, ModeKeys


def sample_element_from_var(all_var):
    result = {}
    for v in all_var:
        try:
            v_rank = len(v.get_shape())
            v_ele1, v_ele2 = v, v
            for j in range(v_rank):
                v_ele1, v_ele2 = v_ele1[0], v_ele2[-1]
            result['sampled/1_%s' + v.name], result['sampled/2_%s' + v.name] = v_ele1, v_ele2
        except:
            pass
    return result


def partial_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


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
            self.logger.info('%s is build in %.4f secs' % (scope_name, time.time() - start_t))

        return f2_v

    return f2


def get_filename(args, mode: ModeKeys):
    return os.path.join(args.result_dir,
                        '-'.join(v for v in [args.model_id, mode.name, args.suffix_output] if
                                 v.strip()) + '.json')


def write_dev_json(f, pred_answers):
    with open(f, 'w', encoding='utf8') as fp:
        for p in pred_answers:
            fp.write(json.dumps(p, ensure_ascii=False) + '\n')


class LossCounter:
    def __init__(self, task_names, log_interval, batch_size, tb_writer):
        self._task_names = task_names
        self._log_interval = log_interval
        self._start_t = time.time()
        self._num_step = 1
        self._batch_size = batch_size
        self._n_steps_loss = 0
        self._n_batch_task_loss = {k: 0.0 for k in self._task_names}
        self._reset_step_loss()
        self._tb_writer = tb_writer
        self._logger = logging.getLogger(APP_NAME)
        self._last_metric = 0

    def _reset_step_loss(self):
        self._last_n_steps_loss = self._n_steps_loss / self._log_interval
        self._n_steps_loss = 0
        self._n_batch_task_loss = {k: 0.0 for k in self._task_names}

    def record(self, fetches):
        self._num_step += 1
        self._n_steps_loss += fetches['loss']
        for k, v in fetches['task_loss'].items():
            self._n_batch_task_loss[k] += v
        if self._trigger():
            self.show_status()
            self._reset_step_loss()
            if self._tb_writer:
                self._tb_writer.add_summary(fetches['merged_summary'], self._num_step)

    def is_overfitted(self, metric):
        if metric - self._last_metric < 1e-6:
            return True
        else:
            self._last_metric = metric
            return False

    def _trigger(self):
        return (self._log_interval > 0) and (self._num_step % self._log_interval == 0)

    def show_status(self):
        cur_loss = self._n_steps_loss / self._log_interval
        self._logger.info('%-4d->%-4d L: %.3f -> %.3f %d/s %s' % (
            self._num_step - self._log_interval + 1, self._num_step,
            self._last_n_steps_loss, cur_loss,
            round(self._num_step * self._batch_size / (time.time() - self._start_t)),
            self._get_multitask_loss_str(self._n_batch_task_loss, normalizer=self._log_interval)
        ))

    @staticmethod
    def _get_multitask_loss_str(loss_dict, normalizer=1.0, show_key=True, show_value=True):
        if show_key and not show_value:
            to_str = lambda x, y: '%s' % x
        elif show_key and show_value:
            to_str = lambda x, y: '%s: %.3f' % (x, y)
        elif not show_key and show_value:
            to_str = lambda x, y: '%.3f' % y
        else:
            to_str = lambda x, y: ''
        return ' '.join([to_str(k, v / normalizer) for k, v in loss_dict.items()])
