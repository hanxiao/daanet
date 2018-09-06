import logging

from tensorflow.python.framework.errors_impl import NotFoundError, InvalidArgumentError

from gpu_env import ModeKeys, APP_NAME
from utils.helper import build_model

logger = logging.getLogger(APP_NAME)


def train(args):
    # check run_mode
    if 'run_mode' in args:
        args.set_hparam('run_mode', ModeKeys.TRAIN.value)
    model = build_model(args)
    try:
        model.restore(use_ema=False, use_partial_loader=False)
        model.reset()  # for continous training, we reset some layers to random if necessary
    except (NotFoundError, InvalidArgumentError) as e:
        logger.debug(e)
        logger.info('no available model, will train from scratch!')

    model.train()


def evaluate(args):
    model = build_model(args)
    model.restore()
    return model.evaluate()


def demo(args):
    args.is_serving = True  # set it to true to ignore data set loading
    model = build_model(args)
    model.restore()
    sample_context = ''
    sample_questions = ['What was Maria Curie the first female recipient of?',
                        'What year was Casimir Pulaski born in Warsaw?',
                        'Who was one of the most famous people born in Warsaw?',
                        'Who was Frédéric Chopin?',
                        'How old was Chopin when he moved to Warsaw with his family?']
    sample_answers = ['Nobel Prize',
                      '1745',
                      'Maria Skłodowska-Curie',
                      'Famous musicians',
                      'seven months old']

    for q, g in zip(sample_questions, sample_answers):
        a = model.predict(sample_context, q)  # real work is here!
        logger.info('QUESTION: %s' % q)
        logger.info('ANSWER: %s <- GOLD: %s' % (a, g))
