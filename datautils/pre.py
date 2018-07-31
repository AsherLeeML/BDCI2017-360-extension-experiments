import sys
sys.path.append('..')

from functools import partial
from WVutils.config import *

from keras.utils import to_categorical
from keras.preprocessing import sequence
from pyltp import SentenceSplitter


import jieba
import logging

jieba.setLogLevel(logging.INFO)
jieba.enable_parallel(4)
jieba.load_userdict('../libs/dict.txt.big')

word_embed_dict = pickle.load(open(Config.word_embed_dict, 'rb'))
word_embed_dict_sp = pickle.load(open(Config.word_embed_dict_sp, 'rb'))

word_unknown = len(word_embed_dict.keys()) + 1

def get_word_seq(contents, word_maxlen=Config.word_seq_maxlen, mode='post', verbose=False):
    word_r = []
    contents = '\n'.join(contents)
    contents = ' '.join(list(jieba.cut(contents))).replace(' \n ', '\n')
    contents = [content.split(' ') for content in contents.split('\n')]
    for content in tqdm(contents, disable=(not verbose)):
        word_c = np.array([word_embed_dict[w] for w in content if w in word_embed_dict])
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    return word_seq

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def batch_generator(contents, labels, batch_size=128, shuffle=True, preprocessfunc=None):
    sample_size=contents.shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_contents = contents[batch_ids]
            batch_contents = preprocessfunc(batch_contents)
            batch_labels = to_categorical(labels[batch_ids])
            yield (batch_contents, batch_labels)
def word_cnn_preprocess(contents, word_maxlen=Config.word_seq_maxlen):
    word_seq = get_word_seq(contents, word_maxlen=word_maxlen)
    return word_seq

def word_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_length=Config.sentence_word_length):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    for index, content in enumerate(contents):
        if index >= len(contents): break
        sentences = SentenceSplitter.split(content)
        word_seq = get_word_seq(sentences, word_maxlen=sentence_length)
        word_seq = word_seq[:sentence_num]
        contents_seq[index][:len(word_seq)] = word_seq
    return contents_seq

def word_cnn_train_batch_generator(train_content, train_label, batch_size=128):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, preprocessfunc=word_cnn_preprocess)

def word_han_train_batch_generator(train_content, train_label, batch_size=128):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, preprocessfunc=word_han_preprocess)

                   
                   