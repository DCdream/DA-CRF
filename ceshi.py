'''
import codecs
from data_utils import create_dico, create_mapping, zero_digits
from loader import char_mapping, tag_mapping
from loader import load_sentences, update_tag_scheme
from loader import augment_with_pretrained, prepare_dataset
import itertools
train_sentences = load_sentences(r"G:\pyworkspace\AS_select_features\ChineseNER-master1\ChineseNER-master\data1\1.train", True, False)
test_sentences = load_sentences(r"G:\pyworkspace\AS_select_features\ChineseNER-master1\ChineseNER-master\data1\1.test", True, False)
dico_chars_train = char_mapping(train_sentences, True)[0]
#训练数据统计的词典
print(dico_chars_train)
dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                r"G:\pyworkspace\AS_select_features\ChineseNER-master1\ChineseNER-master\glove.6B.100d.txt",
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
n_words = len(id_to_char)
for i in range(n_words):
    print(id_to_char[i])
'''
'''
import random
from data_utils import split_train_dev
l = [[['s', 'pos', 3], ['a', 'tn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['v', 'pos', 3], ['a', 'yn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['t', 'pos', 3], ['a', 'un', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['s', 'pos', 3], ['a', 'tn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['v', 'pos', 3], ['a', 'yn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['t', 'pos', 3], ['a', 'un', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['s', 'pos', 3], ['a', 'tn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['v', 'pos', 3], ['a', 'yn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['t', 'pos', 3], ['a', 'un', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['s', 'pos', 3], ['a', 'tn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['v', 'pos', 3], ['a', 'yn', 8], ['k', 'nl', 6], ['c', 'nt', 10]],
     [['t', 'pos', 3], ['a', 'un', 8], ['k', 'nl', 6], ['c', 'nt', 10]], [['t', 'pos', 3], ['a', 'un', 8], ['k', 'nl', 6], ['c', 'nt', 10]]]
random.shuffle(l)
print(l)
print(int(11/10*8+1))
train, dev = split_train_dev(l)
print(len(train))
print(len(dev))
import tensorflow as tf
a = tf.constant([[[], []], [[], []], [[], []]])
import numpy as np
a = np.array([[["1", "2", "3", "9"], [1, 2, 3]], [["4", "5", "6"], [4, 5, 6]], [["1"], [1]]])
sorted_data = sorted(a, key=lambda x: len(x[0]))
print(sorted_data)

import tensorflow as tf
import numpy as np

c = np.random.random([10, 1])
b = tf.nn.embedding_lookup(c, [[1, 3], [2, 4]])

with tf.Session() as sess:
     sess.run(tf.initialize_all_variables())
     print(sess.run(b))
     print(c)
'''
'''
import tensorflow as tf
print(":------------")
a = tf.constant([[[1., 1.], [2., 1.], [2., 3.]], [[1., 3.], [3., 4.], [1., 2.]], [[3., 4.],[2., 2.], [3., 1.]]])
b = tf.constant([[[1., 1.], [2., 1.], [2., 3.]], [[1., 3.], [3., 4.], [1., 2.]], [[3., 4.],[2., 2.], [3., 1.]]])
s = tf.constant([[1., 2., 3.], [2., 1., 1.]])
c = tf.tensordot(a, s, axes=1)
u = tf.constant([2., 3.])
aten = tf.tensordot(a, u, axes=1)
a_b = tf.concat([a, b], axis=-1)

c_1 = tf.expand_dims(c, -1)

char_inputs = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]])
enbedding = tf.nn.embedding_lookup(char_inputs, [[1, 2, 3], [2, 0, 1], [0, 0, 1]])
used = tf.sign(tf.abs(char_inputs))
length = tf.reduce_sum(used, reduction_indices=1)
lengths = tf.cast(length, tf.int32)
with tf.Session() as sess:
     sess.run(tf.initialize_all_variables())
     print(sess.run(c))
     print(sess.run(aten))
     print("a_b---", sess.run(a_b))

     print(sess.run(tf.reshape(a, [-1, 2])))
     print(sess.run(tf.reshape(tf.nn.xw_plus_b(tf.reshape(a, [-1, 2]), s, [0., 0.]), [-1, 3, 2])))
     print(sess.run(tf.nn.softmax(c)))
     print(sess.run(c_1))
     print(sess.run(a*c_1))
     print(a.shape[0])
     print(used)
     print(sess.run(length))
     print(sess.run(lengths))
     print(sess.run(enbedding))
     print(sess.run(tf.concat(enbedding, axis=-1)))
'''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
a = tf.constant([[[1., 1.], [2., 1.], [2., 3.]], [[1., 3.], [3., 4.], [1., 2.]]])
w1 = tf.constant([[1., 1.], [2., 2.]])
w2 = tf.constant([[2., 2.], [2., 2.]])
v = tf.constant([[1.], [2.]])
a_trans = tf.transpose(a, [1, 0, 2])
num_step = a.shape[1]
ai = []
ci = []
for i in range(num_step):
     b = tf.matmul(a_trans[i], w1)
     score = []
     for j in range(num_step):
          c = tf.matmul(a_trans[j], w2)
          d = b + c
          s = tf.matmul(d, v)
          score.append(s)
     score = tf.transpose(score, [1, 0, 2])
     ci_1 = a * score
     ci.append(tf.reduce_sum(ci_1, 1))
     ai.append(score)
ci = tf.transpose(ci, [1, 0, 2])
ai = tf.transpose(ai, [0, 1, 2, 3])
          # ai.append(s)
# ai = tf.transpose(ai, [1, 0, 2])



with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     print("---a:\n", sess.run(a))
     print("a_trains:\n", sess.run(a_trans))
     # print(sess.run(a_trans[0]))
     # print(sess.run(w2 + w1))
     print(sess.run(ai))
     print("---ci_1\n", sess.run(ci_1))

     print(sess.run(ci))