from .bleu_metric.bleu import Bleu
from .exact_f1.exact_f1 import f1_exact_eval
from .meteor.meter import compute_meter_score
from .rouge_metric.rouge import Rouge


def normalize(s):
    """
    Normalize strings to space joined chars.

    Args:
        s: a list of strings.

    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
        "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    f1_exact = f1_exact_eval()

    pred_list, ref_list = [], []
    for k in pred_dict.keys():
        pred_list.append(pred_dict[k][0])
        ref_list.append(ref_dict[k][0])
    f1_score, exact_score = f1_exact.compute_scores(pred_list, ref_list)
    meter_score = compute_meter_score(pred_list, ref_list)
    scores['f1'] = f1_score
    scores['exact'] = exact_score
    scores['meter'] = meter_score
    return scores
