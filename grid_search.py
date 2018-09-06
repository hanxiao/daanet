import itertools
import os
import sys

from ruamel.yaml import YAML

from utils.helper import set_logger, fill_gpu_jobs, get_tmp_yaml


def run():
    logger = set_logger()

    with open('grid.yaml') as fp:
        settings = YAML().load(fp)
        test_set = sys.argv[1:] if len(sys.argv) > 1 else settings['common']['config']
        all_args = [settings[t] for t in test_set]
        entrypoint = settings['common']['entrypoint']
    with open('default.yaml') as fp:
        settings_default = YAML().load(fp)
        os.environ['suffix_model_id'] = settings_default['default']['suffix_model_id']

    cmd = ' '.join(['python app.py', entrypoint, '%s'])

    all_jobs = []
    for all_arg in all_args:
        k, v = zip(*[(k, v) for k, v in all_arg.items()])
        all_jobs += [{kk: pp for kk, pp in zip(k, p)} for p in itertools.product(*v)]
    while all_jobs:
        all_jobs = fill_gpu_jobs(all_jobs, logger,
                                 job_parser=lambda x: cmd % get_tmp_yaml(x,
                                                                         (os.environ['suffix_model_id'] if
                                                                          os.environ['suffix_model_id'] else
                                                                          '+'.join(test_set)) + '-'),
                                 wait_until_next=settings['common']['wait_until_next'],
                                 retry_delay=settings['common']['retry_delay'],
                                 do_shuffle=True)

    logger.info('all jobs are done!')


if __name__ == '__main__':
    run()
