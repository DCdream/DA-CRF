# encoding = utf8
import re
import math
import codecs
import random

import numpy as np
#import jieba
#jieba.initialize()
#jieba.load_userdict('D:\DC\pyproject\data_utils\jieba_dict1.txt')


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    #print(dico)
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    #print("sorted", sorted_items)
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    #print("id_to_item", id_to_item)
    item_to_id = {v: k for k, v in id_to_item.items()}
    #print("item_to_id", item_to_id)
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words

def split_train_dev(train_sentences):
    random.seed(7)
    random.shuffle(train_sentences)
    l = len(train_sentences)
    split_line = int(l / 10 * 8)
    real_train_sentences = train_sentences[0:split_line]
    dev_sentences = train_sentences[split_line:]
    return real_train_sentences, dev_sentences
'''
def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature
'''
def load_lexcion(lexcion_path, nlp):
    print("loading lexcion from {}......".format(lexcion_path))
    l_lexcion = []
    for line in codecs.open(lexcion_path):
        l_lexcion.append(line.strip())
    ll_lexcion = []
    for a_l in l_lexcion:
        ll_lexcion.append(nlp.word_tokenize(a_l))
    l_sorted_lexcion = sorted(ll_lexcion, key=lambda i: len(i), reverse=True)
    print("loading done!")
    return l_sorted_lexcion


def get_lexcion_features(list_strings, l_sorted_lexcion):
    #print("loading lexcion......")
    #nlp = StanfordCoreNLP(r'E:\DC\dataset\泰一指尚评测数据\stanford-corenlp-full-2017-06-09')
    len_list_strings = len(list_strings)
    #l_lexcion = []
    l_lexcion_features = [0] * len_list_strings
    # for line in codecs.open(lexcion_path):
    #     l_lexcion.append(line.strip())
    # l_sorted_lexcion = sorted(l_lexcion, key=lambda i: len(nlp.word_tokenize(i)), reverse=True)
    for a_lex in l_sorted_lexcion:
        #print(type(a_lex))
        if " ".join(a_lex) in " ".join(list_strings) \
                or " ".join(a_lex).lower() in " ".join(list_strings).lower():
        # list_a_lex = nlp.word_tokenize(a_lex)
            len_a_lex = len(a_lex)
            if len_list_strings >= len_a_lex:
                for i in range(len_list_strings):
                    if i <= len_list_strings - len_a_lex:
                        if a_lex == list_strings[i:i + len_a_lex] \
                                or [j.lower() for j in a_lex] == [j.lower() for j in list_strings[i:i + len_a_lex]]:
                            if l_lexcion_features[i:i + len_a_lex] == [0] * len_a_lex:
                                if len_a_lex == 1:
                                    l_lexcion_features[i:i + len_a_lex] = [1] * len_a_lex
                                elif len_a_lex == 2:
                                    l_lexcion_features[i] = 1
                                    l_lexcion_features[i + len_a_lex - 1] = 1
                                elif len_a_lex > 2:
                                    l_lexcion_features[i:i + len_a_lex] = [1] * len_a_lex
                                    l_lexcion_features[i] = 1
                                    l_lexcion_features[i + len_a_lex - 1] = 1
    return l_lexcion_features





def get_pos_ids(poses):
    # jieba共有53种不同的词性
    # all_pos = ['a', 'ad', 'ag', 'an', 'b', 'c', 'd', 'df', 'dg',
    #            'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'mg',
    #            'mq', 'n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt',
    #            'nz', 'o', 'p', 'q', 'r', 'rg', 'rr', 'rz', 's',
    #            't', 'tg', 'u', 'ud', 'ug', 'uj', 'ul', 'uv', 'uz',
    #            'v', 'vd', 'vi', 'vn', 'vq', 'x', 'y', 'z', 'zg', 'eng']
    #stanford共有47种
    all_pos = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
               'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
               'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
               'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
               'WP', 'WP$', 'WRB', '#', '$', ',', '``', "''", '.', ':',
               '-RRB-', '-LRB-', '(', ')', '"']
    pos_ids = []
    for a_pos in poses:
        if a_pos in all_pos:
            pos_ids.append(all_pos.index(a_pos))
        else:
            pos_ids.append(47)
    return pos_ids

def get_dep_ids(dep_name):
    #由语料统计出来，共41个
    all_dep_name = ['nsubj', 'ROOT', 'dobj', 'case', 'nmod', 'cc',
                    'conj', 'xcomp', 'det', 'mwe', 'amod', 'compound',
                    'punct', 'aux', 'advmod', 'neg', 'ccomp', 'mark',
                    'nmod:poss', 'cop', 'acl:relcl', 'nummod', 'acl',
                    'dep', 'appos', 'compound:prt', 'auxpass', 'advcl',
                    'nmod:tmod', 'parataxis', 'nsubjpass', 'discourse',
                    'expl', 'csubj', 'root', 'det:predet', 'nmod:npmod',
                    'iobj', 'cc:preconj', 'CD', 'csubjpass']
    dep_ids = []
    for a_dep in dep_name:
        if a_dep in all_dep_name:
            dep_ids.append(all_dep_name.index(a_dep))
        else:
            dep_ids.append(42)
    return dep_ids

def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    #print(len(new_weights[0]))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cut_to_sentence(text):
    """
    Cut text to sentences
    """
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut=True
            pre_cut=False
        if word in u"。;!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    pre_cut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


class BatchManager(object):

    def __init__(self, data,  batch_size, max_len):
        self.batch_data = self.sort_and_pad(data, batch_size, max_len)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size, max_len):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size], max_len))
        return batch_data

    @staticmethod
    def pad_data(data, max_length):
        strings = []
        chars = []
        #segs = []
        lexcion_teatures = []
        pos_ids = []
        dep_ids = []
        head_ids = []
        targets = []
        # max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, lexcion_feature, pos_id, dep_id, head_id, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            lexcion_teatures.append(lexcion_feature + padding)
            pos_ids.append(pos_id + padding)
            dep_ids.append(dep_id + padding)
            head_ids.append(head_id + padding)
            #segs.append(seg + padding)
            targets.append(target + padding)
        #return [strings, chars, segs, targets]
        return [strings, chars, lexcion_teatures, pos_ids, dep_ids, head_ids, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]



def pad_data(data, max_length):
    strings = []
    chars = []
    lexcion_teatures = []
    pos_ids = []
    dep_ids = []
    head_ids = []
    targets = []
    # paded_data = []
    # max_length = max([len(sentence[0]) for sentence in data])
    for line in data:
        # a_line = []
        string, char, lexcion_feature, pos_id, dep_id, head_id, target = line
        padding = [0] * (max_length - len(string))
        # a_line.append(string + padding)
        # a_line.append(char + padding)
        # a_line.append(lexcion_feature + padding)
        # a_line.append(pos_id + padding)
        # a_line.append(dep_id + padding)
        # a_line.append(head_id + padding)
        # a_line.append(target + padding)
        # paded_data.append(a_line)

        strings.append(string + padding)
        chars.append(char + padding)
        lexcion_teatures.append(lexcion_feature + padding)
        pos_ids.append(pos_id + padding)
        dep_ids.append(dep_id + padding)
        head_ids.append(head_id + padding)
        targets.append(target + padding)
    return [strings, chars, lexcion_teatures, pos_ids, dep_ids, head_ids, targets]
    # return paded_data
'''
def feed_format_data(data):
    strings = []
    chars = []
    lexcion_teatures = []
    pos_ids = []
    dep_ids = []
    head_ids = []
    targets = []
    for line in data:
        string, char, lexcion_feature, pos_id, dep_id, head_id, target = line
        strings.append(string)
        chars.append(char)
        lexcion_teatures.append(lexcion_feature)
        pos_ids.append(pos_id)
        dep_ids.append(dep_id)
        head_ids.append(head_id)
        targets.append(target)
    return [strings, chars, lexcion_teatures, pos_ids, dep_ids, head_ids, targets]
'''
