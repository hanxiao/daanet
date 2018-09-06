import copy
import json
import random
import re


def _tokenize(x):
    tokens = [v for v in re.findall(r"\w+|[^\w]", x, re.UNICODE) if len(v)]  # fix last hanging space
    token_shifts = []
    char_token_map = []
    c, j = 0, 0
    for v in tokens:
        if v.strip():
            token_shifts.append(j)
            j += 1
        else:
            token_shifts.append(-1)
        char_token_map += [token_shifts[-1]] * len(v)
    # remove empty word and extra space in tokens
    tokens = [v.strip() for v in tokens if v.strip()]
    assert len(tokens) == max(char_token_map) + 1, \
        'num tokens must equal to the max char_token_map, but %d vs %d' % (len(tokens), max(char_token_map))
    assert len(char_token_map) == len(x), \
        'length of char_token_map must equal to original string, but %d vs %d' % (len(char_token_map), len(x))
    return tokens, char_token_map


def _char_token_start_end(char_start, answer_text, char_token_map, full_tokens=None):
    # to get the tokens use [start: (end+1)]
    start_id = char_token_map[char_start]
    end_id = char_token_map[char_start + len(answer_text) - 1]
    if full_tokens:
        ans = ' '.join(full_tokens[start_id: (end_id + 1)])
        ans_gold = ' '.join(_tokenize(answer_text)[0])
        assert ans == ans_gold, 'answers are not identical "%s" vs "%s"' % (ans, ans_gold)
    return start_id, end_id


def _dump_to_json(sample):
    return json.dumps(sample).encode()


def _load_from_json(batch):
    return [json.loads(d) for d in batch]


def _parse_line(line):
    return json.loads(line.strip())


def _do_padding(token_ids, token_lengths, pad_id):
    pad_len = max(token_lengths)
    return [(ids + [pad_id] * (pad_len - len(ids)))[: pad_len] for ids in token_ids]


def _do_char_padding(char_ids, token_lengths, pad_id, char_pad_id):
    pad_token_len = max(token_lengths)
    pad_char_len = max(len(xx) for x in char_ids for xx in x)
    pad_empty_token = [char_pad_id] * pad_char_len
    return [[(ids + [pad_id] * (pad_char_len - len(ids)))[: pad_char_len] for ids in x] +
            [pad_empty_token] * (pad_token_len - len(x)) for x in char_ids]


def _dropout_word(x, unk_id, dropout_keep_prob):
    return [v if random.random() < dropout_keep_prob else unk_id for v in x]


def _fast_copy(x, ignore_keys):
    y = {}
    for k, v in x.items():
        if k in ignore_keys:
            y[k] = v
        else:
            y[k] = copy.deepcopy(v)
    return y


def build_vocab(embd_files):
    from utils.vocab import Vocab
    if embd_files[0].endswith('pickle'):
        return Vocab.load_from_pickle(embd_files[0])
    return Vocab(embd_files, lower=True)
