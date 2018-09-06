import random

from base import base_io
from gpu_env import ModeKeys


class DataIO(base_io.BaseDataIO):
    def __init__(self, args):
        super().__init__(args)
        if args.is_serving:
            self.logger.info('model is serving request, ignoring train & dev sets!')
        else:
            self.datasets = {
                ModeKeys.TRAIN: self.load_data(self.args.train_files, ModeKeys.TRAIN),
                ModeKeys.EVAL: self.load_data(self.args.dev_files, ModeKeys.EVAL),
            }
            self.data_pointer = {k: 0 for k in self.datasets.keys()}
            self.post_process_fn = {
                ModeKeys.TRAIN: self.post_process_train,
                ModeKeys.EVAL: self.post_process_eval,
            }
            self.reset_pointer(ModeKeys.TRAIN, shuffle=True)

    def reset_pointer(self, mode, shuffle=False):
        self.data_pointer[mode] = 0
        if shuffle:
            random.shuffle(self.datasets[mode])
            self.logger.info('%s data is shuffled' % mode.name)

    def next_batch(self, batch_size: int, mode: ModeKeys):
        batch = []
        dataset = self.datasets[mode]
        start_pointer = self.data_pointer[mode]
        batch_data = dataset[start_pointer: (start_pointer + batch_size)]
        if len(batch_data) == 0:
            self.reset_pointer(mode, shuffle=(mode == ModeKeys.TRAIN))
            raise EOFError('%s data is exhausted' % mode.name)
        for sample in batch_data:
            batch.append(self.post_process_fn[mode](sample))
            start_pointer += 1
        self.data_pointer[mode] = start_pointer
        return self.make_mini_batch(batch)

    def post_process_train(self, sample):
        """
        # this is important! otherwise you always overwrite the samples
        new_sample = copy.deepcopy(sample)
        # process new sample
        # for example, shuffle dropout.

        return new_sample
        """
        raise NotImplementedError

    def post_process_eval(self, sample):
        return sample
