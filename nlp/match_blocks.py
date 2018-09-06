import tensorflow as tf

from nlp.nn import linear_logit, layer_norm
from nlp.seq2seq.common import dropout, softmax_mask


def Attention_match(context, query, context_mask, query_mask,
                    num_units=None,
                    scope='attention_match_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = context.get_shape().as_list()[-1]
            score = tf.matmul(context, query, transpose_b=True)
        else:
            score = tf.matmul(linear_logit(context, num_units, scope='context2hidden'),
                              linear_logit(query, num_units, scope='query2hidden'),
                              transpose_b=True)
        mask = tf.matmul(tf.expand_dims(context_mask, -1), tf.expand_dims(query_mask, -1), transpose_b=True)
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        masked_score = tf.where(tf.equal(mask, 0), paddings, score / (num_units ** 0.5))  # B, Lc, Lq
        query2context_score = tf.reduce_sum(masked_score, axis=2, keepdims=True)  # B, Lc, 1
        match_score = tf.nn.softmax(query2context_score, axis=1)  # (B, Lc, 1)
        return context * match_score


def Transformer_match(context,
                      query,
                      context_mask,
                      query_mask,
                      num_units=None,
                      num_heads=8,
                      dropout_keep_rate=1.0,
                      causality=True,
                      scope='MultiHead_Attention_Block',
                      reuse=None,
                      residual=False,
                      normalize_output=True,
                      **kwargs):
    """Applies multihead attention.

    Args:
      context: A 3d tensor with shape of [N, T_q, C_q].
      query: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    """
    if num_units is None or residual:
        num_units = context.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units

        # Linear projections
        Q = tf.layers.dense(context, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(query, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(query, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking, aka query
        mask1 = tf.tile(query_mask, [num_heads, 1])  # (h*N, T_k)
        mask1 = tf.tile(tf.expand_dims(mask1, 1), [1, tf.shape(context)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask1, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking  aka, context
        mask2 = tf.tile(context_mask, [num_heads, 1])  # (h*N, T_q)
        mask2 = tf.tile(tf.expand_dims(mask2, -1), [1, 1, tf.shape(query)[1]])  # (h*N, T_q, T_k)
        outputs *= mask2  # (h*N, T_q, T_k)

        # Dropouts
        outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_rate)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        if residual:
            # Residual connection
            outputs += context

        if normalize_output:
            # Normalize
            outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs


def BiDaf_match(a, b, a_mask, b_mask, residual, scope=None, reuse=None, **kwargs):
    # context: [batch, l, d]
    # question: [batch, l2, d]
    with tf.variable_scope(scope, reuse=reuse):
        n_a = tf.tile(tf.expand_dims(a, 2), [1, 1, tf.shape(b)[1], 1])
        n_b = tf.tile(tf.expand_dims(b, 1), [1, tf.shape(a)[1], 1, 1])

        n_ab = n_a * n_b
        n_abab = tf.concat([n_ab, n_a, n_b], -1)

        kernel = tf.squeeze(tf.layers.dense(n_abab, units=1), -1)

        a_mask = tf.expand_dims(a_mask, -1)
        b_mask = tf.expand_dims(b_mask, -1)
        kernel_mask = tf.matmul(a_mask, b_mask, transpose_b=True)
        kernel += (kernel_mask - 1) * 1e5

        con_query = tf.matmul(tf.nn.softmax(kernel, 1), b)
        con_query = con_query * a_mask

        query_con = tf.matmul(tf.transpose(
            tf.reduce_max(kernel, 2, keepdims=True), [0, 2, 1]), a * a_mask)
        query_con = tf.tile(query_con, [1, tf.shape(a)[1], 1])
        if residual:
            return tf.concat([a * query_con, a * con_query, a, query_con], 2)
        else:
            return tf.concat([a * query_con, a * con_query, a, query_con], 2)


def dot_attention(inputs, memory, mask, hidden_size, keep_prob=1.0, is_train=None, scope=None):
    with tf.variable_scope(scope or 'dot_attention'):
        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                tf.layers.dense(d_inputs, hidden_size, use_bias=False, name="inputs"))
            memory_ = tf.nn.relu(
                tf.layers.dense(d_memory, hidden_size, use_bias=False, name="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden_size ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask))
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(tf.layers.dense(d_res, dim, use_bias=False))
            return res * gate
