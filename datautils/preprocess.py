import os

import sys
sys.path.append('..')
from WVutils.config import *
from WVutils.data import *
from sklearn.model_selection import train_test_split
from pre import *

train_data = get_train_data()
train, val = train_test_split(train_data, test_size=0.1)
train.fillna("", inplace=True)
val.fillna("", inplace=True)
train.to_csv(Config.cache_path + 'trainset.csv', index=False, sep='\t')
val.to_csv(Config.cache_path + 'valset.csv', index=False, sep='\t')


train_cnn = word_cnn_preprocess(train.content.values)
val_cnn  = word_cnn_preprocess(val.content.values)
train_han = word_han_preprocess(train.content.values)
val_han = word_han_preprocess(val.content.values)


pickle.dump(train_cnn, open(Config.cache_path + 'train_cnn_seq.pkl', 'wb'))
pickle.dump(val_cnn, open(Config.cache_path + 'val_cnn_seq.pkl', 'wb'))
pickle.dump(train_han, open(Config.cache_path + 'train_han_seq.pkl', 'wb'))
pickle.dump(val_han, open(Config.cache_path + 'val_han_seq.pkl', 'wb'))