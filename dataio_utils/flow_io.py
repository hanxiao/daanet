import json
from typing import List

import tensorflow as tf

from base import base_io
from gpu_env import ModeKeys


# dataio controller
class FlowDataIO(base_io.BaseDataIO):
    def __init__(self, args):
        super().__init__(args)
        if args.is_serving:
            self.logger.info('model is serving request, ignoring train & dev sets!')
        else:
            self.datasets = {
                ModeKeys.TRAIN: self.load_data(self.args.train_files, ModeKeys.TRAIN),
                ModeKeys.EVAL: self.load_data(self.args.dev_files, ModeKeys.EVAL),
            }
            if 'test_files' in self.args:
                self.datasets[ModeKeys.TEST] = self.load_data(self.args.test_files, ModeKeys.TEST)

            self.data_node = {}

    def make_node(self, mode: ModeKeys):
        for k, v in self.datasets.items():
            if k == mode:
                self.data_node[k] = v.make_one_shot_iterator

    def next_batch(self, batch_size: int, mode: ModeKeys):
        return self.data_node[mode]().get_next()

    def load_data(self, file_paths: List[str], mode: ModeKeys):
        dataset = tf.data.TextLineDataset(file_paths) \
            .shuffle(5000) \
            .filter(lambda x: tf.py_func(self._filter_invalid_seq, [x], tf.bool)) \
            .map(lambda x: tf.py_func(self.make_sample, [x], tf.string)) \
            .batch(self.args.batch_size) \
            .map(lambda x: tf.py_func(self.make_mini_batch, [x], tf.string)) \
            .prefetch(self.args.batch_size * 5)

        self.logger.info('loading data for %s' % mode.name)

        return dataset

    def _dump_to_json(self, sample):
        try:
            r = json.dumps(sample).encode()
        except Exception as e:
            print(e)
            r = json.dumps({}).encode()
        return r

    def _load_from_json(self, batch):
        return [json.loads(str(d, encoding='utf8')) for d in batch]

    def _filter_invalid_seq(self, line):
        raise NotImplementedError

    def make_sample(self, line, mode=ModeKeys.TRAIN):
        raise NotImplementedError

    def make_mini_batch(self, data, mode=ModeKeys.TRAIN):
        raise NotImplementedError

    def single2batch(self, context):
        raise NotImplementedError
