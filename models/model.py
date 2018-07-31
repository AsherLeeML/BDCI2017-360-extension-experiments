import os
import tensorflow as tf
import sys
sys.path.append('..')
import keras
import keras.backend as K

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
from recurrentshop import *



class Attention(Layer):
    def __init__(self, 
                 step_dim, 
                 W_regularizer=None, 
                 b_regularizer=None,
                 W_constraint=None, 
                 b_constraint=None, 
                 bias=True, 
                 **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            (input_shape[-1], ),
                          initializer=self.init,
                          name='{}_W'.format(self.name),
                          regularizer=self.W_regularizer,
                          constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        
        if self.bias:
            self.b = self.add_weight(
                (input_shape[1], ),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True
    def compute_mask(self, input, input_mask=None):
        return None
    
    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
def conv_block(data, convs = [3, 4, 5], f = 256, name = 'conv_feat'):
    pools = []
    for c in convs:
        conv = Activation(activation='relu')(BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding='valid')(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

def get_textcnn(seq_length, embed_weight):
    content = Input(shape=(seq_length, ), dtype='int32')
    embedding = Embedding(input_dim=embed_weight.shape[0], weights=[embed_weight], output_dim=embed_weight.shape[1], trainable=False, name='wordEmbedding')
    trans_content = Activation(activation='relu')(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    feat = conv_block(trans_content)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2, activation='softmax')(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_hcnn(sen_num, sen_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sen_length, ), dtype='int32')
    embedding = Embedding(input_dim=embed_weight.shape[0], weights=[embed_weight], output_dim = embed_weight.shape[1], mask_zero=mask_zero, trainable=False, name='Embedding1')
    sen_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sen_embed)
    word_attention = Attention(sen_length)(word_bigru)
    sen_encode = Model(sentence_input, word_attention)
    
    review_input = Input(shape=(sen_num, sen_length), dtype='int32')
    review_encode = TimeDistributed(sen_encode)(review_input)
    feat = conv_block(review_encode)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2, activation='softmax')(fc)
    model = Model(review_input, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def get_han(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length,), dtype="int32")
    embedding = Embedding(
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Model(sentence_input, word_attention)

    review_input = Input(shape=(sent_num, sent_length), dtype="int32")
    review_encode = TimeDistributed(sent_encode)(review_input)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(review_encode)
    sent_attention = Attention(sent_num)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(sent_attention)))
    output = Dense(2,activation="softmax")(fc)
    model = Model(review_input, output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model


def conv_blocks_v2(data, convs=[3, 4, 5], f=256, name='conv_feat'):
    pools = []
    for c in convs:
        conv = Conv1D(f, c, activation='elu', padding='same')(data)
        conv = MaxPooling1D(c)(conv)
        conv = Conv1D(f, c-1, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(c-1)(conv)
        conv = Conv1D(f//2, 2,activation='elu', padding='same')(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Flatten()(conv)
        pools.append(conv)
    return concatenate(pools, name=name)

def get_textcnn_no_TD(seq_length, embed_weight):
    content = Input(shape=(seq_length, ), dtype='int32')
    embedding = Embedding(input_dim=embed_weight.shape[0], weights=[embed_weight], output_dim=embed_weight.shape[1], trainable=False)
    trans_content = Activation(activation='relu')(BatchNormalization()(embedding(content)))
    feat = conv_block(trans_content)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2, activation='softmax')(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_textcnn_v2(seq_length, embed_weight):
    content = Input(shape=(seq_length, ), dtype='int32')
    embedding = Embedding(input_dim=embed_weight.shape[0], weights=[embed_weight], output_dim=embed_weight.shape[1], name='embedding', trainable=False)
    trans_content = Activation(activation='relu')(BatchNormalization()((TimeDisTributed(Dense(256))(embedding(content)))))
    unvec = conv_blocks_v2(trans_content)
    dropfeat = Dropout(0.4)(unvec)
    fc = Dropout(0.4)(Activation(activation='relu')(BatchNormalization()(Dense(300)(dropfeat))))
    output = Dense(2, activation='softmax')(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model