import importlib
import json
import logging
import os
from collections import defaultdict
from math import ceil

import numpy as np
import tensorflow as tf
from ruamel.yaml import YAML

from gpu_env import ModeKeys, APP_NAME, SummaryType
from model_utils.helper import mblock, partial_restore, sample_element_from_var


class BaseModel:
    def __init__(self, args):
        self.logger = logging.getLogger(APP_NAME)
        self.args = args
        self.train_op = None
        self.ema = None
        self.is_var_ema = False
        self.fetch_nodes = defaultdict(lambda: defaultdict(int))
        self.monitored_non_vars = []
        self.sess = None
        self.loss = None
        self._loss = {}  # other auxiliary loss
        self.embed_loaded = False
        dataio = importlib.import_module(args.package_dataio)
        self.data_io = dataio.DataIO(args)
        self.vocab_size = self.data_io.vocab.size()
        self.pretrain_vocab_size = self.data_io.vocab.pretraind_size()
        self.initial_tokens_size = self.data_io.vocab.initial_tokens_size()
        self.vocab_dim = self.data_io.vocab.embed_dim
        try:
            self.char_vocab_size = self.data_io.char_vocab.size()
            self.char_vocab_dim = self.data_io.char_vocab.embed_dim
        except:
            self.char_vocab_size, self.char_vocab_dim = None, None
        self._build_graph()
        if self.args.run_mode == ModeKeys.TRAIN.value:
            self._init_train_op()
            self._init_tensorboard()
            self._set_fetches()
        self.init_session()
        self.write_num_pars()
        if self.args.run_mode == ModeKeys.TRAIN.value:
            self.is_graph_valid()

    def _build_graph(self):
        raise NotImplementedError

    def _set_learning_rate(self):
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

        if self.args.learning_rate_strategy == 'FIXED':
            self.lr = tf.minimum(self.args.learning_rate,
                                 self.args.learning_rate / tf.log(999.) * tf.log(
                                     tf.cast(self.global_step, tf.float32) + 1))
        elif self.args.learning_rate_strategy == 'HALF_COSINE_MAX':
            # from snapshot paper
            t_m = tf.constant(ceil(self.args.learning_rate_reset_epoch * self.args.num_total_samples /
                                   self.args.batch_size), dtype=tf.int32)

            self.lr = (self.args.learning_rate / 2.0) * (
                    tf.cos(tf.constant(3.1415, tf.float32) *
                           tf.cast(tf.mod(self.global_step, t_m), tf.float32)
                           / tf.cast(t_m, tf.float32)) + 1.0)
        elif self.args.learning_rate_strategy == 'HALF_COSINE_ZERO':
            # from snapshot paper
            t_m = tf.constant(ceil(self.args.learning_rate_reset_epoch * self.args.num_total_samples /
                                   self.args.batch_size), dtype=tf.int32)

            self.lr = (self.args.learning_rate / 2.0) * (1.0 -
                                                         tf.cos(tf.constant(3.1415, tf.float32) *
                                                                tf.cast(tf.mod(self.global_step, t_m), tf.float32)
                                                                / tf.cast(t_m, tf.float32)))
        elif self.args.learning_rate_strategy == 'COSINE_ZERO':
            t_m = tf.constant(ceil(self.args.learning_rate_reset_epoch * self.args.num_total_samples /
                                   self.args.batch_size), dtype=tf.int32)

            self.lr = (self.args.learning_rate / 2.0) * (1.0 -
                                                         tf.cos(tf.constant(2 * 3.1415, tf.float32) *
                                                                tf.cast(tf.mod(self.global_step, t_m), tf.float32)
                                                                / tf.cast(t_m, tf.float32)))
        elif self.args.learning_rate_strategy == 'COSINE_MAX':
            t_m = tf.constant(ceil(self.args.learning_rate_reset_epoch * self.args.num_total_samples /
                                   self.args.batch_size), dtype=tf.int32)

            self.lr = (self.args.learning_rate / 2.0) * (1.0 +
                                                         tf.cos(tf.constant(2 * 3.1415, tf.float32) *
                                                                tf.cast(tf.mod(self.global_step, t_m), tf.float32)
                                                                / tf.cast(t_m, tf.float32)))
        elif self.args.learning_rate_strategy == 'COSINE_ZERO_DECAY':
            t_m = tf.constant(ceil(self.args.learning_rate_reset_epoch * self.args.num_total_samples /
                                   self.args.batch_size), dtype=tf.int32)

            self.lr = (self.args.learning_rate /
                       tf.ceil(tf.cast(self.global_step, tf.float32) / tf.cast(t_m, tf.float32)) + 1) \
                      * (1.0 - tf.cos(tf.constant(2 * 3.1415, tf.float32) *
                                      tf.cast(tf.mod(self.global_step, t_m), tf.float32)
                                      / tf.cast(t_m, tf.float32)))
        elif self.args.learning_rate_strategy in ['CYCLE_LINEAR', 'CYCLE_SIN']:
            self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32,
                                      initializer=tf.constant_initializer(self.args.learning_rate),
                                      trainable=False)
        else:
            raise NotImplementedError

    def _set_fetches(self):
        self.add_fetch('loss', self.loss, [ModeKeys.TRAIN])
        self.add_fetch('task_loss', self._loss, [ModeKeys.TRAIN])
        self.add_fetch('_train_op', self.train_op, ModeKeys.TRAIN)
        self.add_fetch('global_step', self.global_step, ModeKeys.TRAIN)
        self.add_fetch('learning_rate', self.lr, ModeKeys.TRAIN)
        if self.args.enable_tensorboard:
            self.add_fetch('merged_summary', tf.summary.merge_all(), [ModeKeys.TRAIN])

    def add_tfboard(self, name, value, mode):
        if self.args.enable_tensorboard:
            if isinstance(mode, list):
                for m in mode:
                    self.add_tfboard(name, value, m)
            elif mode == SummaryType.SCALAR:
                tf.summary.scalar(name, value)
            elif mode == SummaryType.HISTOGRAM:
                tf.summary.histogram(name, value)
            elif mode == SummaryType.SAMPLED:
                self.monitored_non_vars.append(value)
            else:
                raise NotImplementedError

    def add_fetch(self, name, value, mode):
        if isinstance(mode, list):
            for m in mode:
                self.fetch_nodes[m][name] = value
        elif isinstance(mode, ModeKeys):
            self.fetch_nodes[mode][name] = value
        else:
            raise AttributeError('mode must be a list of ModeKeys or a ModeKeys!')

    def get_fetch(self, name, mode):
        return self.fetch_nodes[mode][name]

    def init_session(self):
        # session info
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 10
        config.inter_op_parallelism_threads = 10
        self.sess = tf.Session(config=config)
        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=self.args.saver_max_to_keep)
        self.tb_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph) if \
            self.args.enable_tensorboard else None

    def write_num_pars(self):
        get_num_pars = lambda x: sum(list(map(np.prod, self.sess.run([tf.shape(v) for v in x]))))
        total_num_pars = get_num_pars(tf.trainable_variables())
        total_trained = 0

        group_by_scope = defaultdict(list)
        for v in tf.trainable_variables():
            vscope = v.name.split('/')[0]
            group_by_scope[vscope].append(v)

        for k, v in group_by_scope.items():
            n = get_num_pars(v)
            if k in self.args.fixed_layers:
                self.logger.info('%s%20s : %d' % ('|F|', k, n))
            else:
                self.logger.info('%s%20s : %d' % ('|V|', k, n))
                total_trained += n

        self.logger.info('trainable parameters: %d total: %d' % (total_trained, total_num_pars))

        if 'num_parameters' not in self.args:
            self.args.add_hparam('num_parameters', int(total_num_pars))

    def is_graph_valid(self):
        for v in [self.sess, self.saver, self.loss,
                  self.train_op, self.args, self.logger,
                  self.fetch_nodes, self.data_io]:
            assert v is not None, '%s must be initialized' % v
        self.logger.info('graph passed sanity check!')

    def batch2feed_dict(self, batch, mode):
        # add task-specific learning rate to batch
        batch.update({'ph_dropout_keep_prob': self.args.dropout_keep_params if mode == ModeKeys.TRAIN else 1.0})
        success_binds = []
        ignored_binds = []
        feed_dict = {}
        allv = vars(self)
        for k, v in batch.items():
            if k in allv and isinstance(allv[k], tf.Tensor):
                feed_dict[allv[k]] = v
                success_binds.append(k)
            else:
                ignored_binds.append(k)

        # self.logger.info('success bindings: %s' % success_binds)
        # self.logger.warning('ignored bindings: %s' % ignored_binds)
        return feed_dict

    def run_sess_op(self, batch, mode):
        feed_dict = self.batch2feed_dict(batch, mode)
        return self.sess.run(self.fetch_nodes[mode], feed_dict)

    def is_effective_epoch(self, metric, last_metric, best_metric):
        is_effective = 0
        for k, v in metric.items():
            if v >= best_metric[k]:
                is_effective += 1
        if self.args.early_stop_metric in metric:
            is_key_metric_effective = (metric[self.args.early_stop_metric] > best_metric[self.args.early_stop_metric])
        else:
            is_key_metric_effective = False
        self.logger.info('%d/%d effective metrics! effective %s: %s' % (is_effective, len(last_metric),
                                                                        self.args.early_stop_metric,
                                                                        is_key_metric_effective))

        return is_effective >= (len(best_metric) / 2) or is_key_metric_effective

    def load_embedding(self):
        raise NotImplementedError

    def get_tfboard_vars(self):
        raise NotImplementedError

    def train(self, mode=ModeKeys.TRAIN):
        """
        code example:

        self.load_embedding()
        loss_logger = xxxLoger()

        for j in range(self.args.epoch_last, self.args.epoch_last + self.args.epoch_total + 1):
            self.logger.info('start train epoch %d ...' % j)
            try:
                while True:
                    batch = self.data_io.next_batch(self.args.batch_size, mode)
                    fetches = self.run_sess_op(batch, mode)
                    loss_logger.record(fetches)
            except EOFError:
                self.logger.info('epoch %d is done!' % j)
                metrics = self.evaluate(epoch=j)
                cur_metric = metrics[self.args.metric_early_stop]
                if not loss_logger.is_overfitted(cur_metric):
                    self.save(epoch=j)
                else:
                    self.logger.info('early stop due to overfitting: %s %.4f -> %.4f' % (
                        self.args.metric_early_stop, loss_logger._last_metric, cur_metric))
                    break
        """
        raise NotImplementedError

    def predict(self, inputs, mode=ModeKeys.EVAL):
        """
        inputs : dict
        code example:

        batch = self.data_io.single2batch(context, question)
        fetches = self.run_sess_op(batch, mode)
        s_id, e_id, raw = fetches['start_pos'][0], fetches['end_pos'][0], batch['raw'][0]
        return ' '.join(raw['context'][s_id: (e_id + 1)])
        """
        raise NotImplementedError

    def evaluate(self, epoch=-1, mode=ModeKeys.EVAL):
        """
        code example:

        if epoch < 0:
            epoch = self.args.epoch_best if self.args.epoch_best > 0 else self.args.epoch_last
        preds = {}
        try:
            while True:
                batch = self.data_io.next_batch(self.args.batch_size, mode)
                fetches = self.run_sess_op(batch, mode)
                # process fetches result

        except EOFError:
            self.logger.info('evaluation at epoch %d is done!' % epoch)
            metric = self.save_eval_preds(preds, epoch=epoch)
        return metric
        """
        raise NotImplementedError

    def save_eval_preds(self, preds, epoch=-1, mode=ModeKeys.EVAL):
        """
        save eval result to files

        code example:
        result_file = get_filename(self.args, mode)
        with open(result_file, 'w', encoding='utf8') as fp:
            json.dump(preds, fp, ensure_ascii=False, sort_keys=True)
        self.logger.info('prediction saved to %s' % result_file)
        opt = namedtuple('OPT', 'id epoch pred ref out_file '
                                'na_prob_file na_prob_thresh out_image_dir verbose')(
            self.args.model_id, epoch, result_file, self.args.dev_files[0],
            self.args.out_metric_file, None, 1.0, None, True)
        # evaluate predict result
        # metrics = do_evaluation(opt)
        # log metrics
        for k, v in metrics.items():
            if not k.startswith('_'):
                if isinstance(v, int):
                    self.logger.info('%-20s: %d' % (k, v))
                elif isinstance(v, float):
                    self.logger.info('%-20s: %.4f' % (k, v))
                else:
                    self.logger.info('%-20s: %s' % (k, v))

        self.logger.info('prediction metric is added to %s' % self.args.out_metric_file)

        # save to loss file
        if not os.path.isfile(self.args.loss_csv_file):
            with open(self.args.loss_csv_file, 'w') as fp:
                fp.write('epoch %s\n' % (' '.join(k for k in metrics.keys() if not k.startswith('_'))))

        with open(self.args.loss_csv_file, 'a') as fp:
            fp.write('%d ' % epoch)
            for k, v in metrics.items():
                if not k.startswith('_'):
                    if isinstance(v, int):
                        fp.write('%d ' % v)
                    elif isinstance(v, float):
                        fp.write('%.4f ' % v)
                    else:
                        fp.write('%s ' % v)
            fp.write('\n')
        return metrics
        """
        raise NotImplementedError

    def save_for_serving(self, epoch):
        raise NotImplementedError

    def save(self, epoch):
        self.save_model(epoch)
        self.args.epoch_last = epoch
        self.save_args()

    def save_model(self, epoch, save='save'):
        self.saver.save(self.sess, os.path.join(self.args.save_dir, 'epoch%d' % epoch))
        # tf.train.write_graph(self.sess.graph.as_graph_def(), FLAGS.save, "graph.pb")
        if self.args.save_for_serving:
            self.save_for_serving(epoch)
        self.logger.info('model %sd in %s, at epoch %d' % (save, self.args.save_dir, epoch))

    def save_args(self):
        with open(os.path.join(self.args.save_dir, 'default.yaml'), 'w') as fp:
            YAML().dump(json.loads(self.args.to_json()), fp)

    def restore(self, epoch=-1, use_ema=True, use_partial_loader=False):
        if epoch < 0:
            epoch = self.args.epoch_best if self.args.epoch_best > 0 else self.args.epoch_last

        model_file = os.path.join(self.args.save_dir, 'epoch%d' % epoch)
        if use_partial_loader:
            partial_restore(self.sess, model_file)
            self.logger.info('partial restore variables without EMA!')
        else:
            if (self.ema is None) or (not use_ema):
                self.saver.restore(self.sess, model_file)
                self.is_var_ema = False
                self.logger.info('restore variables without EMA!')
            else:
                variables_to_restore = self.ema.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(self.sess, model_file)
                self.is_var_ema = True
                self.logger.info('EMA variables are restored!')
        self.logger.info('model restored from %s, at epoch %d' % (self.args.save_dir, epoch))

    @mblock('Train_Op')
    def _init_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        self._set_learning_rate()
        all_params = [v for v in tf.trainable_variables() if v.name.split('/')[0] not in self.args.fixed_layers]

        if not all_params:
            self.train_op = tf.no_op()
            self.ema = tf.train.ExponentialMovingAverage(decay=self.args.ema_decay)
            self.logger.warning('No training variables! perform no_op while training')
            return

        with tf.name_scope('Regularization_Layer'):
            if self.args.weight_decay > 0:
                # ref. https://stats.stackexchange.com/questions/29130/
                # difference-between-neural-net-weight-decay-and-learning-rate
                with tf.variable_scope('l2_loss'):
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_params])
                    self.loss += self.args.weight_decay * l2_loss

        if self.args.optim == 'ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        elif self.args.optim == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            optimizer = {
                'RMSP': tf.train.RMSPropOptimizer,
                'ADAGRAD': tf.train.AdagradOptimizer,
            }[self.args.optim](learning_rate=self.args.learning_rate, epsilon=1e-8)

        if self.args.gradient_clip:
            # Calculate and clip gradients
            gradients = tf.gradients(self.loss, all_params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_max_norm)
            train_op = optimizer.apply_gradients(zip(clipped_gradients, all_params),
                                                 global_step=self.global_step)
        else:
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        if self.args.ema_decay > 0:
            self.ema = tf.train.ExponentialMovingAverage(decay=self.args.ema_decay)

            with tf.control_dependencies([train_op]):
                train_op_ema = self.ema.apply(all_params)

            self.train_op = train_op_ema
            self.logger.info('EMA is added to training op!')
        else:
            self.train_op = train_op

        self.all_trainable_vars = all_params

    def _init_tensorboard(self):
        if self.args.enable_tensorboard:
            with tf.name_scope('Basic'):
                self.add_tfboard('learning_rate', self.lr, SummaryType.SCALAR)
                self.add_tfboard('loss', self.loss, SummaryType.SCALAR)
                for k, v in self._loss.items():
                    self.add_tfboard('auxiliary_loss/%s' % k, v, SummaryType.SCALAR)

            with tf.name_scope('Sampled_Vars'):
                sampled = sample_element_from_var(self.all_trainable_vars)
                for k, v in sampled.items():
                    self.add_tfboard(k, v, SummaryType.SCALAR)

            with tf.name_scope('Sampled_NonVars'):
                sampled = sample_element_from_var(self.monitored_non_vars)
                for k, v in sampled.items():
                    self.add_tfboard(k, v, SummaryType.SCALAR)

            other_vars = self.get_tfboard_vars()
            if other_vars is not None:
                for kk, vv in other_vars.items():
                    for k in vv:
                        self.add_tfboard('/'.join([k[1].name.split('/')[0], k[0]]), k[1], kk)

    def reset(self):
        reset_params = [v for v in tf.trainable_variables() if v.name.split('/')[0] in
                        self.args.reset_restored_layers]
        if reset_params:
            total_reset_num_par = sum(list(map(np.prod, self.sess.run([tf.shape(v) for v in reset_params]))))
            self.logger.info('resetting %d parameters from %s layers' %
                             (total_reset_num_par, ','.join(s for s in
                                                            self.args.reset_restored_layers)))
            self.sess.run(tf.variables_initializer(reset_params))
