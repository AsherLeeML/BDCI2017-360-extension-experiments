import os
import sys
sys.path.append('..')
import platform 
import pandas as pd
import numpy as np
import re
import math
import _pickle as pickle
import logging
import time

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S %p', level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler('../log/' + time.strftime('%Y_%m_%d', time.localtime()) + '_logging.log')])

from tqdm import tqdm
from functools import partial 
import warnings

warnings.filterwarnings('ignore')

label2int = {'POSITIVE': 1, 'NEGATIVE':0}
int2label = {1:'POSITIVE', 0:'NEGATIVE'}

class Config():
    data_path = '../data/'
    cache_path = data_path + 'cache/'
    
    train_file = data_path + 'train.tsv'
    word_embed_dict = cache_path + 'word_embed_.dict.pkl'
    word_embed_dict_sp = cache_path + 'word_embed_sp.dict.pkl'
    word_embed_weight_path = cache_path + 'word_embed_.npy'
    word_embed_weight_path_sp = cache_path + 'word_embed_sp.npy'
    
    sentence_num = 45
    sentence_word_length = 48
    sentence_char_length = 84
    
    word_seq_maxlen = 1000