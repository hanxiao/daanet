import sys

from gpu_env import DEVICE_ID, MODEL_ID, CONFIG_SET
from utils.helper import set_logger, parse_args, get_args_cli


def run():
    set_logger(model_id='%s:%s' % (DEVICE_ID, MODEL_ID))
    followup_args = get_args_cli(sys.argv[3:]) if len(sys.argv) > 3 else None
    args = parse_args(sys.argv[2] if len(sys.argv) > 2 else None, MODEL_ID, CONFIG_SET, followup_args)
    getattr(__import__('api'), sys.argv[1])(args)


if __name__ == '__main__':
    run()
