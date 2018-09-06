"""Sequence-to-Sequence with attention model.
"""

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from gpu_env import ModeKeys, SummaryType
from model_utils.helper import mblock
from nlp.encode_blocks import LSTM_encode, CNN_encode
from nlp.match_blocks import dot_attention, Transformer_match
from nlp.nn import get_var, highway_network, linear_logit
from nlp.seq2seq.pointer_generator import PointerGeneratorDecoder, \
    PointerGeneratorBahdanauAttention, PointerGeneratorAttentionWrapper
from nlp.seq2seq.rnn import multi_rnn_cell
from .base import RCBase


# import numpy as np
# import seq2seq_lib


class RCCore(RCBase):
    def __init__(self, args):
        super().__init__(args)

    def _build_graph(self):
        self._placeholders()
        self._shortcuts()
        self.loss = -1
        self._masks()
        self._embed()
        self._encode()
        self._decode()
        if self.args.run_mode == ModeKeys.TRAIN.value:
            self._model_loss()
        # if self.args.run_mode != 'decode':

    @mblock('Input_Layer')
    def _placeholders(self):
        # passage token input
        self.ph_passage = tf.placeholder(tf.int32, [None, None], name="passage")
        self.ph_passage_chars = tf.placeholder(tf.int32, [None, None, None], name="passage_chars")
        # question token input
        self.ph_question = tf.placeholder(tf.int32, [None, None], name="question")
        self.ph_question_chars = tf.placeholder(tf.int32, [None, None, None], name="question_chars")
        # answer
        self.ph_answer = tf.placeholder(tf.int32, [None, None], name="answer")  # answer token input
        self.ph_answer_chars = tf.placeholder(tf.int32, [None, None, None], name="answer_chars")
        # length
        self.ph_passage_length = tf.placeholder(tf.int32, [None], name="passage_length")
        self.ph_question_length = tf.placeholder(tf.int32, [None], name="question_length")
        self.ph_answer_length = tf.placeholder(tf.int32, [None], name="answer_length")

        # max number of oov words in this batch
        self.ph_max_oov_length = tf.placeholder(
            tf.int32,
            shape=[],
            name='source_oov_words')
        # input tokens using source oov words and vocab
        self.ph_passage_extend_tokens = tf.placeholder(
            tf.int32,
            shape=[None, None],
            name='source_extend_tokens')

        self.ph_q_decode_input = tf.placeholder(
            tf.int32, [None, None],
            name="question_decode_input")
        self.ph_q_decode_target = tf.placeholder(
            tf.int32, [None, None],
            name="question_decode_target")
        self.ph_q_decode_length = tf.placeholder(
            tf.int32, [None],
            name="question_decode_length")

        self.ph_a_decode_input = tf.placeholder(
            tf.int32, [None, None],
            name="answer_decode_input"
        )
        self.ph_a_decode_target = tf.placeholder(
            tf.int32, [None, None],
            name="answer_decode_target"
        )
        self.ph_a_decode_length = tf.placeholder(
            tf.int32, [None],
            name="answer_decode_length"
        )

        self.ph_a_start_label = tf.placeholder(
            tf.int32, [None],
            name="answer_start_id")
        self.ph_a_end_label = tf.placeholder(tf.int32, [None], name="answer_end_id")

        if self.args.use_answer_masks:
            self.ph_answer_masks = tf.placeholder(
                tf.int32, [None, None, None],
                name="answer_masks")

        self.ph_dropout_keep_prob = tf.placeholder(
            tf.float32,
            name='dropout_keep_prob')
        self.ph_word_emb = tf.placeholder(tf.float32,
                                          [self.pretrain_vocab_size, self.vocab_dim],
                                          name='word_embed_mat')
        self.ph_char_emb = tf.placeholder(tf.float32,
                                          [self.char_vocab_size, self.char_vocab_dim],
                                          name='char_embed_mat')
        self.ph_tokenid_2_charsid = tf.placeholder(tf.int32,
                                                   [self.vocab_size, self.args.max_token_len],
                                                   name='ph_tokenid_2_charids')
        self.ph_is_train = tf.placeholder(tf.bool, [])

    def _shortcuts(self):
        self.batch_size = tf.shape(self.ph_passage)[0]
        self.max_p_len = tf.shape(self.ph_passage)[1]
        self.max_q_len = tf.shape(self.ph_question)[1]
        self.max_a_len = tf.shape(self.ph_answer)[1]
        self.max_q_char_len = tf.shape(self.ph_question_chars)[2]
        self.max_p_char_len = tf.shape(self.ph_passage_chars)[2]
        self.max_a_char_len = tf.shape(self.ph_answer_chars)[2]

    @mblock('Input_Layer/Mask')
    def _masks(self):
        self.a_mask = tf.sequence_mask(
            self.ph_answer_length, tf.shape(self.ph_answer)[1],
            dtype=tf.float32,
            name='answer_mask')

        self.p_mask = tf.sequence_mask(
            self.ph_passage_length, tf.shape(self.ph_passage)[1],
            dtype=tf.float32,
            name='passage_mask')

        self.q_mask = tf.sequence_mask(
            self.ph_question_length, tf.shape(self.ph_question)[1],
            dtype=tf.float32,
            name='question_mask')

        self.decode_q_mask = tf.sequence_mask(
            self.ph_q_decode_length, tf.shape(self.ph_q_decode_target)[1],
            dtype=tf.float32,
            name='decode_question_mask')
        self.decode_a_mask = tf.sequence_mask(
            self.ph_a_decode_length, tf.shape(self.ph_a_decode_target)[1],
            dtype=tf.float32,
            name='decode_answer_mask'
        )

    @mblock('Embedding')
    def _embed(self):
        self.pretrained_word_embeddings = get_var(
            'pretrained_word_embeddings',
            shape=[self.pretrain_vocab_size, self.vocab_dim],
            trainable=self.args.embed_trainable)

        self.init_tokens_embeddings = tf.get_variable(name="init_tokens_embeddings",
                                                      shape=[self.initial_tokens_size - 1, self.vocab_dim],
                                                      initializer=tf.random_normal_initializer())
        self.pad_tokens_embeddings = tf.get_variable(name="pad_tokens_embeddings",
                                                     shape=[1, self.vocab_dim],
                                                     initializer=tf.zeros_initializer(), trainable=False)
        self.pretrain_word_embed_init = self.pretrained_word_embeddings.assign(self.ph_word_emb)
        # "unk, start end, pad" in the end of embeddings
        self.word_embeddings = tf.concat(
            [self.pretrained_word_embeddings, self.init_tokens_embeddings, self.pad_tokens_embeddings], axis=0,
            name='word_embeddings')
        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.ph_dropout_keep_prob)
        self.char_emb = get_var('char_embeddings', shape=[self.char_vocab_size, self.args.char_embed_size],
                                trainable=True)
        self.tokenid_2_charsid_map = tf.get_variable('tokenid_2_charsid_map', dtype=tf.int32,
                                                     shape=[self.vocab_size, self.args.max_token_len],
                                                     trainable=False, initializer=tf.zeros_initializer())
        self.tokenid_2_charsid_map_init = self.tokenid_2_charsid_map.assign(self.ph_tokenid_2_charsid)
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE) as scope:
            self.embedding_scope = scope

            def emb_ff(ids):
                """ 
                :param ids:  shape of ids is [batch] or [batch,L]
                :return: embedding [batch, D] or [batch, L, D]
                """
                num_of_dim = ids.get_shape().ndims
                if num_of_dim == 1:
                    ids = tf.reshape(ids, [self.batch_size, 1])
                condition = tf.less(ids, self.vocab_size)
                ids = tf.where(condition, ids, tf.ones_like(ids) * self.data_io.unk_id)
                max_axis1_len = tf.shape(ids)[-1]
                char_ids = tf.nn.embedding_lookup(self.tokenid_2_charsid_map, ids)  # B,L,max_token_len
                max_axis2_len = tf.shape(char_ids)[-1]
                token_emb = tf.nn.embedding_lookup(self.word_embeddings, ids)
                char_emb = tf.reshape(tf.nn.embedding_lookup(self.char_emb, char_ids),  # B,L,max_token_len,D_char
                                      [self.batch_size * max_axis1_len, max_axis2_len, self.args.char_embed_size])
                char_emb = CNN_encode(char_emb, filter_size=self.args.embed_filter_size,
                                      num_filters=self.args.char_embed_size, scope=scope, reuse=tf.AUTO_REUSE)
                char_emb = tf.reshape(tf.reduce_max(char_emb, axis=1),
                                      [self.batch_size, max_axis1_len, self.args.char_embed_size])
                concat_emb = tf.concat([token_emb, char_emb], axis=-1)
                concat_emb = linear_logit(concat_emb, self.args.embedding_output_dim, scope=scope, reuse=tf.AUTO_REUSE)

                highway_out = highway_network(concat_emb, self.args.highway_layer_num, scope=scope, reuse=tf.AUTO_REUSE)
                if num_of_dim == 1:
                    return tf.squeeze(highway_out, axis=1)  # B,D
                else:
                    return highway_out  # B,L,D

        self.embedding_func = emb_ff
        self.p_emb = self.embedding_func(self.ph_passage)
        self.q_emb = self.embedding_func(self.ph_question)
        self.a_emb = self.embedding_func(self.ph_answer)

    @mblock('Encoding')
    def _encode(self):
        with tf.variable_scope('Passage_Encoder'):
            self.p_encodes_rnn = LSTM_encode(self.p_emb, num_layers=self.args.encode_num_layers,
                                             num_units=self.args.encode_num_units, direction=self.args.encode_direction,
                                             scope='p_encode')

            all_p_encodes = [self.p_encodes_rnn]
            if self.args.self_attention_encode:
                self.p_encodes_trans = Transformer_match(self.p_emb, self.p_emb, self.p_mask, self.p_mask,
                                                         self.args.self_attention_num_units,
                                                         scope='Passage_Encoder_trans')
                all_p_encodes.append(self.p_encodes_trans)
            if self.args.highway_encode:
                self.p_encodes_highway = highway_network(self.p_emb, 1, num_units=self.args.highway_num_units,
                                                         scope='Passage_Encoder')
                all_p_encodes.append(self.p_encodes_highway)

            self.p_encodes = tf.concat(all_p_encodes, -1) * tf.expand_dims(self.p_mask, -1)

        with tf.variable_scope("Question_Encoder") as q_encode_scope:
            self.q_encodes_rnn = LSTM_encode(self.q_emb, num_layers=self.args.encode_num_layers,
                                             num_units=self.args.encode_num_units, direction=self.args.encode_direction,
                                             scope='q_encode')

            all_q_encodes = [self.q_encodes_rnn]
            if self.args.self_attention_encode:
                self.q_encodes_trans = Transformer_match(self.q_emb, self.q_emb, self.q_mask, self.q_mask,
                                                         self.args.self_attention_num_units,
                                                         scope=q_encode_scope, layer_norm_scope='2', causality=True)
                all_q_encodes.append(self.q_encodes_trans)

            if self.args.highway_encode:
                self.q_encodes_highway = highway_network(self.q_emb, num_layers=1,
                                                         num_units=self.args.highway_num_units, scope=q_encode_scope)
                all_q_encodes.append(self.q_encodes_highway)

            self.q_encodes = tf.concat(all_q_encodes, -1) * tf.expand_dims(self.q_mask, -1)

            def question_encoder_f(inputs):

                all_e = []
                if self.args.share_transformer_encode:
                    fake_q_mask = tf.ones(shape=tf.shape(inputs)[:2], dtype=tf.float32)
                    all_e.append(
                        Transformer_match(inputs, inputs, fake_q_mask, fake_q_mask, self.args.self_attention_num_units,
                                          scope=q_encode_scope, reuse=True, layer_norm_scope='2', causality=True))
                if self.args.share_highway_encode:
                    all_e.append(highway_network(inputs, 1, num_units=self.args.highway_num_units, scope=q_encode_scope,
                                                 reuse=True))
                all_e = tf.concat(all_e, axis=-1)
                return all_e[:, -1, :]  # B,D

            self.question_encoder_func = question_encoder_f

        with tf.variable_scope("Answer_Encoder") as a_encode_scope:
            self.a_encodes_rnn = LSTM_encode(self.a_emb, num_layers=self.args.encode_num_layers,
                                             num_units=self.args.encode_num_units, direction=self.args.encode_direction,
                                             scope='a_encode')

            all_a_encodes = [self.a_encodes_rnn]
            if self.args.self_attention_encode:
                self.a_encodes_trans = Transformer_match(self.a_emb, self.a_emb, self.a_mask, self.a_mask,
                                                         self.args.self_attention_num_units,
                                                         scope=a_encode_scope, layer_norm_scope='2', causality=True)
                all_a_encodes.append(self.a_encodes_trans)
            if self.args.highway_encode:
                self.a_encodes_highway = highway_network(self.a_emb, 1, num_units=self.args.highway_num_units,
                                                         scope=a_encode_scope)
                all_a_encodes.append(self.a_encodes_highway)

            self.a_encodes = tf.concat(all_a_encodes, -1) * tf.expand_dims(self.a_mask, -1)

            def answer_encoder_f(inputs):
                all_e = []
                if self.args.share_transformer_encode:
                    fake_q_mask = tf.ones(shape=tf.shape(inputs)[:2], dtype=tf.float32)
                    all_e.append(
                        Transformer_match(inputs, inputs, fake_q_mask, fake_q_mask, self.args.self_attention_num_units,
                                          scope=a_encode_scope, reuse=True, layer_norm_scope='2',
                                          causality=True))

                if self.args.share_highway_encode:
                    all_e.append(highway_network(inputs, 1, num_units=self.args.highway_num_units, scope=a_encode_scope,
                                                 reuse=True))
                enc_res = tf.concat(all_e, -1)

                return enc_res[:, -1, :]  # B,D

            self.answer_encoder_func = answer_encoder_f

        self.encode_dim = self.args.encode_num_units * 2
        with tf.variable_scope("Question_Passage_Attention"):
            self.qp_att = dot_attention(self.q_encodes, self.p_encodes,
                                        mask=self.p_mask,
                                        hidden_size=self.encode_dim,
                                        keep_prob=self.args.dropout_keep_prob,
                                        is_train=self.ph_is_train,
                                        scope="question_attention")  # B, LQ, D

        with tf.variable_scope("Answer_Passage_Attention"):
            self.ap_att = dot_attention(self.a_encodes, self.p_encodes,
                                        mask=self.p_mask,
                                        hidden_size=self.encode_dim,
                                        keep_prob=self.args.dropout_keep_prob,
                                        is_train=self.ph_is_train,
                                        scope="question_attention")  # B, LQ, D

        with tf.variable_scope("Question_Passage_Encode"):
            self.question_encoder_cell = multi_rnn_cell(
                "LSTM", self.args.decoder_num_units,
                is_train=self.args.run_mode == ModeKeys.TRAIN.value,
                keep_prob=self.args.dropout_keep_prob,
                num_layers=self.args.lstm_num_layers)

            _, self.qp_encoder_state = tf.nn.dynamic_rnn(
                cell=self.question_encoder_cell,
                inputs=self.qp_att,
                sequence_length=self.ph_question_length,
                dtype=tf.float32)

            self.qp_encoder_outputs = dot_attention(
                self.p_encodes, self.qp_att,
                mask=self.q_mask,
                hidden_size=self.encode_dim,
                keep_prob=self.args.dropout_keep_prob,
                is_train=self.ph_is_train,
                scope="passage_attention")

        with tf.variable_scope("Answer_Passage_Encode"):
            self.answer_encoder_cell = multi_rnn_cell(
                "LSTM", self.args.decoder_num_units,
                is_train=self.args.run_mode == ModeKeys.TRAIN.value,
                keep_prob=self.args.dropout_keep_prob,
                num_layers=self.args.lstm_num_layers)

            _, self.ap_encoder_state = tf.nn.dynamic_rnn(
                cell=self.answer_encoder_cell,
                inputs=self.ap_att,
                sequence_length=self.ph_answer_length,
                dtype=tf.float32)

            self.ap_encoder_outputs = dot_attention(
                self.p_encodes, self.ap_att,
                mask=self.a_mask,
                hidden_size=self.encode_dim,
                keep_prob=self.args.dropout_keep_prob,
                is_train=self.ph_is_train,
                scope="passage_attention")

        self.encode_dim = self.args.final_projection_num

        self.add_tfboard('passage_encode', self.p_encodes,
                         SummaryType.HISTOGRAM)
        self.add_tfboard('question_encode', self.q_encodes,
                         SummaryType.HISTOGRAM)
        self.add_tfboard('qp_encoder_state', self.qp_encoder_state,
                         SummaryType.HISTOGRAM)
        self.add_tfboard('qp_encoder_outputs', self.qp_encoder_outputs,
                         SummaryType.HISTOGRAM)
        self.add_tfboard('ap_encoder_state', self.ap_encoder_state,
                         SummaryType.HISTOGRAM)
        self.add_tfboard('ap_encoder_outputs', self.ap_encoder_outputs,
                         SummaryType.HISTOGRAM)

    @mblock('Decoder')
    def _decode(self):
        vsize = self.vocab_size
        with tf.variable_scope("decoder_output"):
            projection_layer = layers_core.Dense(units=vsize, use_bias=False)  # use_bias
            answer_decoder_cell, answer_initial_state = self.get_decode_cell_state(self.qp_encoder_outputs,
                                                                                   self.qp_encoder_state,
                                                                                   encoder_func=self.answer_encoder_func,
                                                                                   scope="answer_decoder_cell_state")
            question_decoder_cell, question_initial_state = self.get_decode_cell_state(self.ap_encoder_outputs,
                                                                                       self.ap_encoder_state,
                                                                                       encoder_func=self.question_encoder_func,
                                                                                       scope="question_decoder_cell_state")

            if self.args.run_mode == ModeKeys.TRAIN.value:
                # answer decoder
                answer_training_decoder = self.get_training_decoder(self.ph_a_decode_input, self.ph_a_decode_length,
                                                                    answer_decoder_cell, answer_initial_state,
                                                                    projection_layer,
                                                                    embedding_func=self.embedding_func,
                                                                    scope="answer_training_decoder")
                question_training_decoder = self.get_training_decoder(self.ph_q_decode_input, self.ph_q_decode_length,
                                                                      question_decoder_cell, question_initial_state,
                                                                      projection_layer,
                                                                      embedding_func=self.embedding_func,
                                                                      scope="question_training_decoder")

                # Training decoding
                # answer
                self.answer_decoder_outputs, self.answer_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=answer_training_decoder,
                    impute_finished=True,
                    scope="answer_decoder")
                self.answer_decoder_logits = self.answer_decoder_outputs.rnn_output
                self.add_fetch('answer_decoder_logits', self.answer_decoder_logits, [ModeKeys.TRAIN])

                # question
                self.question_decoder_outputs, self.question_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=question_training_decoder,
                    impute_finished=True,
                    scope="question_decoder")
                self.question_decoder_logits = self.question_decoder_outputs.rnn_output
                self.add_fetch('question_decoder_logits', self.question_decoder_logits, [ModeKeys.TRAIN])

            else:
                answer_inference_decoder = self.get_inference_decoder(answer_decoder_cell, answer_initial_state,
                                                                      projection_layer,
                                                                      embedding_func=self.embedding_func,
                                                                      scope="answer_inference_decoder")
                question_inference_decoder = self.get_inference_decoder(question_decoder_cell, question_initial_state,
                                                                        projection_layer,
                                                                        embedding_func=self.embedding_func,
                                                                        scope="question_inference_decoder")
                # Inference Decoding
                # Answer
                self.answer_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=answer_inference_decoder,
                    maximum_iterations=100,
                    impute_finished=False,
                    scope="answer_decoder")

                self.answer_decoder_logits = self.answer_decoder_outputs.sample_id  # B, L
                self.add_fetch('answer_decoder_logits', self.answer_decoder_logits, [ModeKeys.DECODE, ModeKeys.EVAL])
                # Question
                self.question_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=question_inference_decoder,
                    maximum_iterations=100,
                    impute_finished=False,
                    scope="question_decoder")

                self.question_decoder_logits = self.question_decoder_outputs.sample_id  # B, L
                self.add_fetch('question_decoder_logits', self.question_decoder_logits,
                               [ModeKeys.DECODE, ModeKeys.EVAL])

    @mblock('Loss')
    def _model_loss(self):
        qa_loss = self._loss_calc_helper(self.ph_a_decode_length, self.ph_a_decode_target, self.decode_a_mask,
                                         self.answer_decoder_logits, self.answer_decoder_state)
        qg_loss = self._loss_calc_helper(self.ph_q_decode_length, self.ph_q_decode_target, self.decode_q_mask,
                                         self.question_decoder_logits, self.question_decoder_state)
        if self.args.task_name == 'qa':
            self.loss = qa_loss
        elif self.args.task_name == 'qg':
            self.loss = qg_loss
        else:
            self.loss = qa_loss + qg_loss
        self._loss['loss'] = self.loss
        self._loss['qa_loss'] = qa_loss
        self._loss['qg_loss'] = qg_loss

    def load_embedding(self):
        if self.args.embed_use_pretrained and not self.embed_loaded:
            self.sess.run([self.pretrain_word_embed_init, self.tokenid_2_charsid_map_init],
                          feed_dict={
                              self.ph_word_emb: self.data_io.vocab.pretrained_embeddings,
                              self.ph_tokenid_2_charsid: self.data_io.tokenid2charsid,
                          })
            self.embed_loaded = True

    def get_tfboard_vars(self):
        return {
            SummaryType.HISTOGRAM: [
                ('answer_decoder_logits', self.answer_decoder_logits),
                ('question_decoder_logits', self.question_decoder_logits),
            ]
        }

    def get_training_decoder(self, decoder_inputs, decoder_length, decoder_cell, train_initial_state, projection_layer,
                             embedding_func, scope='training_decoder', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            decoder_embedding_inputs = embedding_func(decoder_inputs)
            training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding_inputs, decoder_length)
            training_decoder = PointerGeneratorDecoder(
                source_extend_tokens=self.ph_passage_extend_tokens,
                source_oov_words=self.ph_max_oov_length,
                coverage=self.args.use_coverage,
                cell=decoder_cell,
                helper=training_helper,
                initial_state=train_initial_state,
                output_layer=projection_layer)
            return training_decoder

    def get_inference_decoder(self, decoder_cell, train_initial_state, projection_layer, embedding_func,
                              scope='inference_decoder', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            start_tokens = tf.tile(
                tf.constant([self.data_io.start_token_id],
                            dtype=tf.int32), [self.batch_size])

            # using greedying decoder right now
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=embedding_func,
                start_tokens=start_tokens,
                end_token=self.data_io.stop_token_id)

            inference_decoder = PointerGeneratorDecoder(
                source_extend_tokens=self.ph_passage_extend_tokens,
                source_oov_words=self.ph_max_oov_length,
                coverage=self.args.use_coverage,
                cell=decoder_cell,
                helper=helper,
                initial_state=train_initial_state,
                output_layer=projection_layer)
            return inference_decoder

    def get_decode_cell_state(self, encoder_output, encoder_state, encoder_func=None, scope='decode_cell_state',
                              reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            _cell = multi_rnn_cell(
                "LSTM", self.args.decoder_num_units,
                is_train=self.args.run_mode == ModeKeys.TRAIN.value,
                keep_prob=self.args.dropout_keep_prob,
                num_layers=self.args.lstm_num_layers)
            enc_lengths = self.ph_passage_length
            attention_mechanism = PointerGeneratorBahdanauAttention(
                self.encode_dim, encoder_output,
                memory_sequence_length=enc_lengths,
                coverage=self.args.use_coverage)

            decoder_cell = PointerGeneratorAttentionWrapper(
                cell=_cell,
                encoder_func=encoder_func,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.args.decoder_num_units,
                alignment_history=True,
                coverage=self.args.use_coverage)

            initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)
            initial_state = initial_state.clone(cell_state=encoder_state)
            return decoder_cell, initial_state

    def _loss_calc_helper(self, decode_length, decode_target, decode_mask, decoder_logits, decoder_state):
        max_dec_len = tf.reduce_max(
            decode_length,
            name="max_dec_len")
        # targets: [batch_size x max_dec_len]
        # this is important, because we may have padded endings
        targets = tf.slice(decode_target, [
            0, 0
        ], [-1, max_dec_len], 'targets')

        i1, i2 = tf.meshgrid(tf.range(self.batch_size),
                             tf.range(max_dec_len),
                             indexing="ij")
        indices = tf.stack((i1, i2, targets), axis=2)
        probs = tf.gather_nd(decoder_logits, indices)

        # To prevent padding tokens got 0 prob, and get inf when calculating log(p), we set the lower bound of prob
        # I spent a lot of time here to debug the nan losses, inf * 0 = nan
        probs = tf.where(tf.less_equal(probs, 0),
                         tf.ones_like(probs) * 1e-10, probs)
        crossent = -tf.log(probs)

        loss = tf.reduce_sum(
            crossent * decode_mask) / tf.to_float(self.batch_size)

        if self.args.use_coverage:
            # we got all the alignments from last state
            # shape is: batch * atten_len * max_len
            alignment_history = tf.transpose(decoder_state.alignment_history.stack(), [1, 2, 0])
            coverage_loss = tf.minimum(alignment_history, tf.cumsum(
                alignment_history,
                axis=2,
                exclusive=True))
            coverage_loss = self.args.coverage_loss_weight * \
                            tf.reduce_sum(coverage_loss / tf.to_float(self.batch_size))
            loss += coverage_loss
        return loss
