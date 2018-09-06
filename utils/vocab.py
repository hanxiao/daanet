import logging
import pickle

import numpy as np

from gpu_env import APP_NAME


class Vocab:

    @staticmethod
    def load_from_pickle(fp):
        with open(fp, 'rb') as fin:
            return pickle.load(fin)

    def __init__(self, embedding_files, lower=True):
        self.logger = logging.getLogger(APP_NAME)
        self.id2token = {}
        self.token2id = {}
        self.lower = lower

        # pretrain里面可能有<unk>, 自定义的unk设成<_unk_>防止冲突。
        self.pad_token = '<_pad_>'
        self.unk_token = '<_unk_>'
        self.start_token = '<_start_>'
        self.stop_token = '<_stop_>'
        self.initial_tokens = [self.unk_token, self.start_token, self.stop_token, self.pad_token]

        self.embed_dim = 0
        self.embeddings = None
        self.pretrained_embeddings = None  # 存储预训练向量
        self.initial_tokens_embedding = None

        if embedding_files is not None:
            for w in embedding_files:
                self.load_pretrained(w)

    def size(self):
        return len(self.id2token)

    def pretraind_size(self):
        if self.pretrained_embeddings is not None:
            return self.pretrained_embeddings.shape[0]
        return 0

    def initial_tokens_size(self):
        return len(self.initial_tokens)

    def get_id(self, token, fallback_chars=False):
        token = token.lower() if self.lower else token
        if fallback_chars:
            return self.token2id.get(token, self.token2id[self.unk_token] if len(token) == 1 else [self.get_id(c) for
                                                                                                   c in token])
        else:
            return self.token2id.get(token, self.token2id[self.unk_token])

    def get_token(self, idx):
        return self.id2token.get(idx, self.unk_token)

    def get_token_with_oovs(self, idx, oovs):
        token = self.get_token(idx)
        if idx >= self.size() and oovs is not None:
            idx = idx - self.size()
            try:
                token = oovs[idx]
            except Exception as e:
                token = self.unk_token
        return token

    def add(self, token):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        return idx

    def load_pretrained(self, embedding_path):
        self.logger.info('loading word embedding from %s' % embedding_path)
        trained_embeddings = {}
        num_line = 0
        valid_dim = None
        with open(embedding_path, 'r', encoding='utf8') as fin:
            for line in fin:
                contents = line.strip().split()
                if len(contents) == 0:
                    continue
                token = contents[0]
                num_line += 1
                if valid_dim and len(contents) != valid_dim + 1:
                    self.logger.debug('bad line: %d in embed files!' % num_line)
                    continue

                trained_embeddings[token] = list(map(float, contents[1:]))
                if valid_dim is None:
                    valid_dim = len(contents) - 1

        # rebuild the token x id map
        if not self.token2id:
            self.logger.info('building token-id map...')
            for token in trained_embeddings.keys():
                self.add(token)
            for token in self.initial_tokens:  # initial tokens 放在后面
                if token in trained_embeddings:
                    raise NameError('initial tokens "%s" in pretraind embedding!' % token)
                self.add(token)
        else:
            self.logger.info('use existing token-id map')

        # load inits tokens
        self.initial_tokens_embedding = np.zeros([len(self.initial_tokens), valid_dim])
        # load pretrained embeddings
        embeddings = np.zeros([len(trained_embeddings), valid_dim])
        for token in trained_embeddings.keys():
            embeddings[self.get_id(token)] = trained_embeddings[token]

        self.pretrained_embeddings = embeddings  # 存储预训练向量
        self.embeddings = np.concatenate([embeddings, self.initial_tokens_embedding], axis=0)  # 所有emb

        self.embed_dim = self.embeddings.shape[1]

        self.logger.info('size of embedding %d x %d' % (self.embeddings.shape[0], self.embeddings.shape[1]))
        self.logger.info('size of pretrain embedding %d x %d' % (
            self.pretrained_embeddings.shape[0], self.pretrained_embeddings.shape[1]))
        # self.logger.info('first 3 lines:  %s', embeddings[2:5, :])
        # self.logger.info('last 3 lines:  %s', embeddings[-3:, :])

    def tokens2ids(self, tokens):
        return [self.get_id(x) for x in tokens]

    def tokens2ids_with_oovs(self, tokens, init_oovs=[], dynamic_oovs=True):
        # oovs = []
        ids = []
        ids_with_oov = []
        oov_dict = {}
        if not dynamic_oovs:
            oov_dict = {v: i for i, v in enumerate(init_oovs)}
        for x in tokens:
            id = self.get_id(x)
            ids.append(id)
            lx = x.lower()
            if id == self.get_id(self.unk_token):
                if x.lower() in oov_dict:
                    id = self.size() + oov_dict[lx]
                elif dynamic_oovs:
                    oov_dict[x.lower()] = len(oov_dict)
                    id = self.size() + oov_dict[lx]
            ids_with_oov.append(id)
        oovs = [0] * len(oov_dict)
        for k, v in oov_dict.items():
            oovs[v] = k
        return ids, ids_with_oov, oovs
