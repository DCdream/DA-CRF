# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn1
from utils import result_to_json
from data_utils import create_input, iobes_iob

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

from Attention import attention_layer
#from Attention1 import attention_layer1

class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.attention_dim = config["attention_dim"]
        self.gru_dim = config["gru_dim"]
       # self.seg_dim = config["seg_dim"]
        self.pos_dim = config["pos_dim"]
        self.dep_name_dim = config["dep_name_dim"]
        self.dependency_dim = config["dependency_dim"]
        self.lexcion_dim = config["lexcion_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]

        self.max_len = config["max_len"]

        #self.num_segs = 4
        self.num_lexcion_features = 5
        self.num_poses = 48
        self.num_deps = 42

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        #shape = [batch_size, max_len]
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.lexcion_feature_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="LexcionFeatureInputs")
        # self.seg_inputs = tf.placeholder(dtype=tf.int32,

        #                                  name="SegInputs")
        self.pos_id_inputs = tf.placeholder(dtype=tf.int32,
                                            shape=[None, None],
                                            name="PosIdInputs")
        self.dep_id_inputs = tf.placeholder(dtype=tf.int32,
                                            shape=[None, None],
                                            name="DepIdInputs")
        self.head_id_inputs = tf.placeholder(dtype=tf.int32,
                                            shape=[None, None],
                                            name="HeadIdInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        self.mask = tf.cast(self.char_inputs, tf.bool)
        # print("lengths-----", self.lengths)

        # print("maxlen-----", self.max_len)
        # print("dropout-----", self.dropout.eval())
        # print("num_step---", self.num_steps)

        # embeddings for chinese character and segmentation representation
        # embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        embedding = self.embedding_layer(self.char_inputs, self.lexcion_feature_inputs, self.pos_id_inputs, self.dep_id_inputs, self.head_id_inputs, config)

        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)
       # dep_inputs = tf.nn.dropout(dep_embedding, self.dropout)
        print(lstm_inputs)
        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        attention1_outputs,_ = attention_layer(lstm_outputs, self.num_steps, self.max_len, self.attention_dim, self.gru_dim, self.lengths)


        #attention1_outputs, _, SCORES = attention_layer1(lstm_outputs, self.num_steps, self.max_len, self.attention_dim, self.gru_dim, self.lengths)
       # attention1_outputs = attention_layer1_with_dep(lstm_outputs, dep_inputs, self.mask, self.num_steps, self.max_len, self.attention_dim, self.gru_dim, self.lengths)

        # update_att_outputs1 = update_attention_outputs_layer1(attention1_outputs, self.gru_dim, self.lengths)
        # print(lstm_outputs)
        # lstm_outputs = tf.nn.dropout(lstm_outputs, self.dropout)

       # attention1_outputs = self.attention_layer1(lstm_outputs, self.attention_dim)
        # print("attention_outputs", attention_outputs)
        # attention1_outputs = tf.nn.dropout(attention1_outputs, self.dropout)
       # updated_attention1_outputs = self.update_attention_outputs_layer1(attention1_outputs, self.gru_dim, self.lengths)
        # updated_attention1_outputs = tf.nn.dropout(updated_attention1_outputs, self.dropout)
       # attention2_outputs = self.attention_layer2(lstm_outputs, updated_attention1_outputs, self.attention_dim)
        # attention2_outputs = tf.nn.dropout(attention2_outputs, self.dropout)
       # updated_attention2_outputs = self.update_attention_outputs_layer2(attention2_outputs, self.gru_dim, self.lengths)
        # attention_outputs = tf.nn.dropout(attention_outputs, self.dropout)
       # attention3_outputs = self.attention_layer2(lstm_outputs, updated_attention2_outputs, self.attention_dim, name="attention3")
       # updated_attention3_outputs = self.update_attention_outputs_layer2(attention3_outputs, self.gru_dim, self.lengths, name="update3")
       # attention4_outputs = self.attention_layer2(lstm_outputs, updated_attention3_outputs, self.attention_dim, name="attention4")
       # updated_attention4_outputs = self.update_attention_outputs_layer2(attention4_outputs, self.gru_dim, self.lengths, name="update4")

       # self.att_s = tf.slice(SCORES, [0, 0, 0, 0], [20, 1, 1, 20])

        # logits for tags
        self.logits = self.project_layer(attention1_outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, lexcion_feature_inputs, pos_id_inputs, dep_id_inputs, head_id_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """

        embedding = []
       # dep_embedding = []
        # shape = [batch_size, max_len, embedding_dim]
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["lexcion_dim"]:
                print("Using Lexcion......")
                # shape = [batch_size, max_len, lex_dim]
                with tf.variable_scope("lexcion_embedding"), tf.device('/cpu:0'):
                    self.lexcion_loookup = tf.get_variable(
                        name="lexcion_embedding",
                        shape=[self.num_lexcion_features,self.lexcion_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.lexcion_loookup, lexcion_feature_inputs))
            if config["pos_dim"]:
                print("Using PosTags.......")
                #shape = [batch_size, max_len, pos_dim]
                with tf.variable_scope("pos_embedding"), tf.device('/cpu:0'):
                    self.pos_id_lookup = tf.get_variable(
                        name="pos_embedding",
                        shape=[self.num_poses, self.pos_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.pos_id_lookup, pos_id_inputs))
            if config["dep_name_dim"]:
                print("Using dep_name......")
                # shape = [batch_size, max_len, dep_name_dim]
                with tf.variable_scope("dep_name_embedding"), tf.device('/cpu:0'):
                    self.dep_id_lookup = tf.get_variable(
                        name="dep_name_embedding",
                        shape=[self.num_deps, self.dep_name_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.dep_id_lookup, dep_id_inputs))
            if config["dependency_dim"]:
                print("Using dep........")
               # dep_embedding.append(tf.nn.embedding_lookup(self.char_lookup, head_id_inputs))
                # shape = [batch_size, max_len, dep_dim]
                #print(self.dependency_dim)
                embedding.append(tf.nn.embedding_lookup(self.char_lookup, head_id_inputs))
                #print(len(embedding))
            embed = tf.concat(embedding, axis=-1)
           # dep_embeded = tf.concat(dep_embedding, axis=-1)
        print(embed)
       # print("depembedding----", dep_embeded)
        # shape = [batch_size, max_len, (embedding+pos_dim_dep_name_dim+dep_dim组合)]
        #return embed, dep_embeded
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn1.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
            # shape = [batch_size, max_len, 2*lstm_dim]
        #这里之后可以加attention层
        print(tf.concat(outputs, axis=2))
        return tf.concat(outputs, axis=2)

    def attention_layer1(self, lstm_output, attention_size, name=None):
        #lstm_output, shape=[batch_size, max_len, 2*lstm_dim]
        #attention_size
        hidden_size = lstm_output.shape[-1]
        print("hidden_size----", hidden_size)
        with tf.variable_scope("Attention1" if not name else name):
            with tf.variable_scope("Attention_matrix1"):
                W = tf.get_variable("W", shape=[hidden_size, attention_size],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[attention_size],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
                u = tf.get_variable("u", shape=[attention_size],
                                    dtype=tf.float32, initializer=self.initializer)
                #shape = [batch, max_len, attention_size]
                attentioned = tf.reshape(tf.tanh(tf.tensordot(lstm_output, W, axes=1) + b), [-1, self.num_steps, attention_size])
                print("attentioned---", attentioned)
                #shape= [batch, max_len]
                attention_score = tf.reshape(tf.tensordot(attentioned, u, axes=1), [-1, self.num_steps])
                print("attention_score---", attention_score)
                # shape= [batch, max_len]
                normalized_attention_score = tf.nn.softmax(attention_score)
                # shape= [batch, max_len, 2*lstm_dim]
                attention_output = lstm_output * tf.expand_dims(normalized_attention_score, -1)
                print("attention_output---",attention_output)
        return attention_output

    def update_attention_outputs_layer1(self, attention_outputs, hidden_units, lengths, name=None):
        with tf.variable_scope("update_attention_outputs1" if not name else name):
            gru_cell = rnn_cell.GRUCell(hidden_units)
            outputs, state = rnn.dynamic_rnn(gru_cell, attention_outputs, dtype=tf.float32, sequence_length=lengths)
            #shape = [b, t, 2d]
            return outputs

    def attention_layer2(self, lstm_output, attention1_outs, attention_size, name=None):
        attention2_input = tf.concat([lstm_output, attention1_outs], axis=-1)
        hidden_size = attention2_input.shape[-1]
        with tf.variable_scope("Attention2" if not name else name):
            with tf.variable_scope("Attention_matrix2"):
                W = tf.get_variable("W", shape=[hidden_size, attention_size],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[attention_size],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
                u = tf.get_variable("u", shape=[attention_size],
                                    dtype=tf.float32, initializer=self.initializer)
                # shape = [batch, max_len, attention_size]
                attentioned = tf.reshape(tf.tanh(tf.tensordot(attention2_input, W, axes=1) + b),
                                         [-1, self.num_steps, attention_size])
                # print("attentioned---", attentioned)
                # shape= [batch, max_len]
                attention_score = tf.reshape(tf.tensordot(attentioned, u, axes=1), [-1, self.num_steps])
                # print("attention_score---", attention_score)
                # shape= [batch, max_len]
                normalized_attention_score = tf.nn.softmax(attention_score)
                # shape= [batch, max_len, 2*lstm_dim]
                attention_output = lstm_output * tf.expand_dims(normalized_attention_score, -1)
                # print("attention_output---", attention_output)
        return attention_output

    def update_attention_outputs_layer2(self, attention_outputs, hidden_units, lengths, name=None):
        with tf.variable_scope("update_attention_outputs2" if not name else name):
            gru_cell = rnn_cell.GRUCell(hidden_units)
            outputs, state = rnn.dynamic_rnn(gru_cell, attention_outputs, dtype=tf.float32, sequence_length=lengths)
            #shape = [b, t, 2d]
            return outputs

    def project_layer(self, last_layer_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param last_layer_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                #shape = [batch_size*max_len, 2embedding_dim]
                output = tf.reshape(last_layer_outputs, shape=[-1, self.lstm_dim*2])
                print("project_out----", output)
                #这边对bilstm的输出做了一个XW + b, shape = [batch_size*max_len, embedding_dim]
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
                print("hidden--", hidden)
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                #这边能否用softmax，这边用softmax是否比用
                #shape = [batch_size*max_len, num_tags(BIO为3, BEMSO为5)]
                pred = tf.nn.xw_plus_b(hidden, W, b)
                print("pre----", pred)
                print(tf.reshape(pred, [-1, self.num_steps, self.num_tags]))
            #shape = [batch_size, max_len, num_tags]
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1]),
                 small * tf.ones(shape=[self.batch_size, 1, 1])], axis=-1)
            end_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                 small * tf.ones(shape=[self.batch_size, 1, 1]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)

            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 2]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits, end_logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets,
                 tf.cast((self.num_tags + 1) * tf.ones([self.batch_size, 1]), tf.int32)], axis=-1)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                # transition_params=self.trans,
                sequence_lengths=lengths + 2)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        # _, chars, segs, tags = batch
        # feed_dict = {
        #     self.char_inputs: np.asarray(chars),
        #     self.seg_inputs: np.asarray(segs),
        #     self.dropout: 1.0,
        # }
        _, chars, lexcion_features, pos_ids, dep_ids, head_ids, tags = batch
        # print(type(len(chars[0])))
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.lexcion_feature_inputs: np.asarray(lexcion_features),
            self.pos_id_inputs: np.asarray(pos_ids),
            self.dep_id_inputs: np.asarray(dep_ids),
            self.head_id_inputs: np.asarray(head_ids),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits


    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0, small]])
        end = np.asarray([[small] * self.num_tags + [small, 0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 2])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits, end], axis=0)
            # print('logits shape:', logits.shape)
            # print('matrix shape:', matrix.shape)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:len(path) - 1])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    #result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        #return tags
        #print(inputs[0][0])
        return result_to_json(inputs[0][0], tags)
