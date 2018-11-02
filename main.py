# encoding=utf8
import os
import codecs
import csv
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_as
from data_utils import load_word2vec, create_input, input_from_line, BatchManager, load_lexcion, split_train_dev, pad_data
#from stanfordcorenlp import StanfordCoreNLP
import random

flags = tf.app.flags
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
# configurations for the model
#flags.DEFINE_integer("seg_dim",     0,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("pos_dim",     0,         "Embedding size of pos, 0 if not used")
flags.DEFINE_integer("dep_name_dim", 0,       "Embedding size of dep_name, 0 if not used")
flags.DEFINE_integer("dependency_dim", 0,    "Embedding size of dep, 0 if not used")
flags.DEFINE_integer("lexcion_dim",    0,      "Embedding size of lexcion, 0 if not used")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_integer("attention_dim", 200,      "Attention_dim")
flags.DEFINE_integer("gru_dim",       100,        "Gru hidden units")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 30,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "glove.6B.100d.txt", "Path for pre_trained embedding")
flags.DEFINE_string("lexcion_file", os.path.join("lexcion", "restaurant15_dict.csv"), "Path for lexcion file")
flags.DEFINE_string("train_file",   os.path.join("data1", "restaurant16_train_POS_DEP_BIO_data.csv"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data1", "restaurant16_test_POS_DEP_BIO_data.csv"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data1", "restaurant16_test_POS_DEP_BIO_data.csv"),   "Path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id, max_len):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["pos_dim"] = FLAGS.pos_dim
    config["dep_name_dim"] = FLAGS.dep_name_dim
    config["dependency_dim"] = FLAGS.dependency_dim
    config["lexcion_dim"] = FLAGS.lexcion_dim
    config["num_tags"] = len(tag_to_id)
    # config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["attention_dim"] = FLAGS.attention_dim
    config["gru_dim"] = FLAGS.gru_dim
    config["batch_size"] = FLAGS.batch_size

    config["lexcion_file"] = FLAGS.lexcion_file

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower

    config["max_epoch"] = FLAGS.max_epoch
    config["max_len"] = max_len
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    as_results = model.evaluate(sess, data, id_to_tag)
   # logger.info(att_scores)
    eval_lines = test_as(as_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
   # train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
   # dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    all_train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    train_sentences, dev_sentences = split_train_dev(all_train_sentences)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # update_tag_scheme(dev_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
           # dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars_train = char_mapping(all_train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(all_train_sentences, FLAGS.lower)
           # _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(all_train_sentences)
       # _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

   # nlp = StanfordCoreNLP(r'E:\DC\dataset\泰一指尚评测数据\stanford-corenlp-full-2017-06-09')
    #l_sorted_lexcion = load_lexcion(FLAGS.lexcion_file, nlp)
    l_sorted_lexcion = []
    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, l_sorted_lexcion, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, l_sorted_lexcion, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, l_sorted_lexcion, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    max_len = max([len(sentence[0]) for sentence in train_data + test_data + dev_data])

    train_manager = BatchManager(train_data, FLAGS.batch_size, max_len)
    dev_manager = BatchManager(dev_data, 800, max_len)
    test_manager = BatchManager(test_data, 800, max_len)

    # random.shuffle(train_data)


    # pad_test_data = pad_data(test_data)
    # pad_dev_data = pad_data(dev_data)

    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id, max_len)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(FLAGS.max_epoch):
            random.shuffle(train_data)
            pad_train_data = pad_data(train_data, max_len)
            strings, chars, lexcion_teatures, pos_ids, dep_ids, head_ids, targets = pad_train_data
            for j in range(0, len(strings), FLAGS.batch_size):
                batch = [strings[j: j + FLAGS.batch_size],
                         chars[j: j + FLAGS.batch_size],
                         lexcion_teatures[j: j + FLAGS.batch_size],
                         pos_ids[j: j + FLAGS.batch_size],
                         dep_ids[j: j + FLAGS.batch_size],
                         head_ids[j: j + FLAGS.batch_size],
                         targets[j: j + FLAGS.batch_size]]
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "AS loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger, i)
                evaluate(sess, model, "test", test_manager, id_to_tag, logger)
        evaluate(sess, model, "test", test_manager, id_to_tag, logger)

def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)







def main(_):

    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()
    # clean(FLAGS)
    # train()


if __name__ == "__main__":
    tf.app.run(main)



