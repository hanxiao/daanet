import json
import os

import tensorflow as tf

from base import base_model
from gpu_env import ModeKeys
from model_utils.helper import LossCounter, get_filename
from utils.eval_4.eval import compute_bleu_rouge
from utils.helper import build_model


# model controller
class RCBase(base_model.BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def train(self, mode=ModeKeys.TRAIN):
        self.load_embedding()
        loss_logger = LossCounter(self.fetch_nodes[mode]['task_loss'].keys(),
                                  log_interval=self.args.log_interval,
                                  batch_size=self.args.batch_size,
                                  tb_writer=self.tb_writer)
        pre_metric = {self.args.metric_early_stop: 0.0}
        for j in range(self.args.epoch_last + 1, self.args.epoch_last + self.args.epoch_total + 1):
            self.logger.info('start train epoch %d ...' % j)
            try:
                while True:
                    batch = self.data_io.next_batch(self.args.batch_size, mode)
                    fetches = self.run_sess_op(batch, mode)
                    loss_logger.record(fetches)
            except EOFError:
                self.save(j)
                metric = self.restore_evaluate(self.args)
                # if metric[self.args.metric_early_stop] < pre_metric[self.args.metric_early_stop]:
                #    self.logger.info('early stop in epoch %s' % j)
                #    break
                pre_metric = metric
                self.logger.info('epoch %d is done!' % j)

    def restore_evaluate(self, args):
        args.set_hparam('run_mode', ModeKeys.EVAL.value)
        args.set_hparam('dropout_keep_prob', 1.0)
        graph = tf.Graph()
        with graph.as_default():
            model = build_model(args, False)
            model.restore()
            return model.evaluate()

    def evaluate(self, epoch=-1, mode=ModeKeys.EVAL):
        if epoch < 0:
            epoch = self.args.epoch_best if self.args.epoch_best > 0 else self.args.epoch_last
        a_pred_dict = {}
        a_ref_dict = {}
        q_pred_dict = {}
        q_ref_dict = {}
        try:
            while True:
                batch = self.data_io.next_batch(self.args.batch_size, mode)
                fetches = self.run_sess_op(batch, mode)
                batch_a_pred_dict, batch_a_ref_dict, batch_q_pred_dict, batch_q_ref_dict = self.parse_result(batch,
                                                                                                             fetches)
                a_pred_dict.update(batch_a_pred_dict)
                a_ref_dict.update(batch_a_ref_dict)
                q_pred_dict.update(batch_q_pred_dict)
                q_ref_dict.update(batch_q_ref_dict)
        except EOFError:
            qa_metric = compute_bleu_rouge(a_pred_dict, a_ref_dict)
            qg_metric = compute_bleu_rouge(q_pred_dict, q_ref_dict)
            qa_metric['type'] = 'qa'
            qg_metric['type'] = 'qg'
            self.save_metrics(qa_metric, epoch=epoch)
            self.save_metrics(qg_metric, epoch=epoch)
            self.save_eval_preds({"pred_dict": a_pred_dict, "ref_dict": a_ref_dict, 'type': "qa"}, epoch=epoch)
            self.save_eval_preds({"pred_dict": q_pred_dict, "ref_dict": q_ref_dict, 'type': "qg"}, epoch=epoch)
            self.logger.info('evaluation at epoch %d is done!' % epoch)
            # self.logger.info(metric)
        return qa_metric

    def predict(self, inputs, mode=ModeKeys.EVAL):
        raise NotImplementedError

    def save_eval_preds(self, preds, epoch=-1, mode=ModeKeys.EVAL):
        result_file = get_filename(self.args, mode)
        with open(result_file, 'w', encoding='utf8') as fp:
            json.dump(preds['pred_dict'], fp, ensure_ascii=False, sort_keys=True)
        self.logger.info('sample preds')
        sample_count = 20
        p_type = preds['type']
        for qid, pred in preds['pred_dict'].items():
            if sample_count < 0:
                break
            ans = preds['ref_dict'][qid]
            if p_type == 'qa':
                self.logger.info("qid=%s" % qid)
                self.logger.info("answer=%s" % ans)
                self.logger.info("pred_answer=%s" % pred)
            else:
                self.logger.info("qid=%s" % qid)
                self.logger.info("question=%s" % ans)
                self.logger.info("pred_question=%s" % pred)
            sample_count -= 1

    def save_metrics(self, metrics, epoch=-1):
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

    def parse_result(self, batch, fetches):
        def get_pred_and_ture(logits, true_tokens, oovs):
            pred = []
            for tid in logits:
                if tid != self.data_io.stop_token_id:
                    pred.append(self.data_io.vocab.get_token_with_oovs(tid, oovs))
                else:
                    break
            pred = " ".join(pred)
            true = " ".join(true_tokens)
            return pred, true

        a_pred_dict = {}
        a_ref_dict = {}
        q_pred_dict = {}
        q_ref_dict = {}
        for qid, ans, questions, a_logits, q_logits, oov_tokens in zip(batch['qid'], batch['answer_tokens'],
                                                                       batch['question_tokens'],
                                                                       fetches['answer_decoder_logits'],
                                                                       fetches['question_decoder_logits'],
                                                                       batch['oovs']):
            a_pred, a_true = get_pred_and_ture(a_logits, ans, oov_tokens)
            a_pred_dict[qid] = [a_pred]
            a_ref_dict[qid] = [a_true]

            q_pred, q_true = get_pred_and_ture(q_logits, questions, oov_tokens)
            q_pred_dict[qid] = [q_pred]
            q_ref_dict[qid] = [q_true]
        return a_pred_dict, a_ref_dict, q_pred_dict, q_ref_dict
