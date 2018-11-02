import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

INF = 1e30

def attention_layer(att_inputs, num_step, max_len, attention_size, gru_hidden, lengths, name=None):
    #att_inputs, att的输入, [batch_size, max_len, dim], 在建图是batch_size和max_len都是None
    #attention_size, att之后，输出的维度
    #max_len是通过placeholder传入的句子最长长度，应为每个batch的句子最长长度不同，所以这个max_len以feed——dict传入
    print("attinput-----", att_inputs)
    print("maxlen-----", max_len)
   # hidden_size = att_inputs.shape[-1]
    # max_len1 = att_inputs.shape[1]
    # print(max_len1)

    #att_inputs_tranpose = tf.transpose(att_inputs, [1, 0, 2])
    #Ct = []
    with tf.variable_scope("Attention_layer1" if not name else name):
        hidden_size = att_inputs.shape[-1]
        att_inputs_tranpose = tf.transpose(att_inputs, [1, 0, 2])
        Ct = []
       # St = []
        with tf.variable_scope("Attention_compute1"):
            w1 = tf.get_variable("w1", shape=[hidden_size, attention_size],
                                 dtype=tf.float32, initializer=initializers.xavier_initializer())
            w2 = tf.get_variable("w2", shape=[hidden_size, attention_size],
                                 dtype=tf.float32, initializer=initializers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[attention_size],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            b2 = tf.get_variable("b2", shape=[attention_size],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            u = tf.get_variable("u", shape=[attention_size, 1],
                                dtype=tf.float32, initializer=initializers.xavier_initializer())

            input_w1 = tf.reshape(tf.tensordot(att_inputs, w1, axes=1) + b1, [-1, num_step, attention_size])
            for t in range(max_len):
                slice_w2 = tf.matmul(att_inputs_tranpose[t], w2) + b2
                add_input_jt = tf.tanh(tf.expand_dims(slice_w2, 1) + input_w1)
                score_a_step = tf.reshape(tf.tensordot(add_input_jt, u, axes=1), [-1, 1, num_step])
                normalized_score = tf.nn.softmax(score_a_step)
               # St.append(normalized_score)
               # Ct.append(tf.matmul(normalized_score, att_inputs))
                normalize_s_a_step = tf.reshape(normalized_score, [-1, num_step, 1])
                Ct.append(tf.reduce_sum(att_inputs * normalize_s_a_step, 1))
            C = tf.transpose(Ct, [1, 0, 2])
           # S = tf.transpose(St, [0, 1, 2, 3])
           # C = tf.concat(Ct, axis=1)
            print(C)
        with tf.variable_scope("gate"):
            concat_C_att_input = tf.concat([att_inputs, C], axis=-1)
            g_dim = concat_C_att_input.get_shape().as_list()[-1]
            w = tf.get_variable("w", shape=[g_dim, g_dim],
                                dtype=tf.float32, initializer=initializers.xavier_initializer())
            gate = tf.nn.sigmoid(tf.reshape(tf.tensordot(concat_C_att_input, w, axes=1), [-1, num_step, g_dim]))
            gated = gate * concat_C_att_input
           # gru_cell = rnn_cell.GRUCell(gru_hidden)
           # gru_cell = rnn_cell.LSTMCell(gru_hidden)
            gru_cell = {}
            for direction in ["forward", "backword"]:
                with tf.variable_scope(direction):
                    gru_cell[direction] = rnn_cell.GRUCell(gru_hidden)
            outputs, state = rnn.bidirectional_dynamic_rnn(gru_cell["forward"], gru_cell["backword"], gated, dtype=tf.float32, sequence_length=lengths)
            outputs = tf.concat(outputs, axis=2)
           # outputs, state = rnn.dynamic_rnn(gru_cell, gated, dtype=tf.float32, sequence_length=lengths)
            print("attouts---", outputs)
    return outputs, state
'''
def update_attention_outputs_layer1(att_outputs, hidden_units, lengths, name=None):
    with tf.variable_scope("update_attention_outputs1" if not name else name):
        with tf.variable_scope("gate"):
            gru_cell = rnn_cell.GRUCell(hidden_units)
            outputs, state = rnn.dynamic_rnn(gru_cell, att_outputs, dtype=tf.float32, sequence_length=lengths)
        return outputs
'''

def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val
