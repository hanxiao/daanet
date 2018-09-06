import os
import re
import sys

import tensorflow as tf

from gpu_env import MODEL_ID

MODEL_PATH = './ext'
print(sys.path)


class Predictor:

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        print("Loading model..., please wait!", flush=True)
        self.models, self.graphs, self.model_ids = load_models(MODEL_PATH)
        print("Load finished!", flush=True)

    def predict(self, model_id, inputs):
        """
        model_id: model dir name
        content : str
        questions: list
        """
        answers = ["no matching model"]
        for i, mid in enumerate(self.model_ids):
            if mid == model_id:
                model = self.models[i]
                answer = model.predict(inputs)
                break
        return answer


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, cur_dir)

    __import__(mod_str)
    sys.path.remove(cur_dir)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def delete_module(modname):
    from sys import modules
    del_keys = []
    for mod_key, mod_value in modules.items():
        if modname in mod_key:
            del_keys.append(mod_key)
        elif modname in str(mod_value):
            del_keys.append(mod_key)

    for key in del_keys:
        del modules[key]


def load_models(save_path):
    model_ids = list_models(save_path)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    print('all available models: %s' % model_ids, flush=True)
    all_models = []
    all_graphs = [tf.Graph() for _ in range(len(model_ids))]
    for m, g in zip(model_ids, all_graphs):
        yaml_path = os.path.join(save_path, m, '%s.yaml' % m)
        args = parse_args(yaml_path, MODEL_ID)
        # args.del_hparam('is_serving')
        if args.get('is_serving') is None:
            args.add_hparam('is_serving', True)
        args.set_hparam('is_serving', True)
        with g.as_default():
            model_path = cur_dir + '/%s' % (m)
            sys.path.insert(0, model_path)
            dync_build = import_class("%s.utils.helper.build_model" % (m))
            model = dync_build(args, reset_graph=False)
            model.restore()
            all_models.append(model)
            print('model %s is loaded!' % m, flush=True)
            sys.path.remove(model_path)
            delete_module(m)

    return all_models, all_graphs, model_ids


def parse_args(yaml_path, model_id):
    from tensorflow.contrib.training import HParams
    from ruamel.yaml import YAML

    hparams = HParams()
    hparams.add_hparam('model_id', model_id)

    with open(yaml_path) as fp:
        customized = YAML().load(fp)
        for k, v in customized.items():
            if k in hparams:
                hparams.set_hparam(k, v)
            else:
                hparams.add_hparam(k, v)
    return hparams


def list_models(save_path):
    model_ids = list(filter(lambda x: os.path.isdir(os.path.join(save_path, x))
                                      and bool(re.match('[0-9]*-[0-9]*', x)), os.listdir(save_path)))
    return model_ids
