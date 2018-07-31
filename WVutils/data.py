import sys

sys.path.append('..')

from WVutils.config import *
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import jieba
jieba.load_userdict('../libs/dict.txt.big')

def get_data(data_path, label=True):
    data = []
    data_error_rows = []
    with open(data_path, 'r') as p:
        f = p.readlines()
    
    for line in f:
        segs = line.strip().split('\t')
        if len(segs) == (4 if label else 3):
            row = {}
            row['title'] = segs[1]
            row['content'] = segs[2]
            if label: row['label'] = label2int[segs[3]]
            data.append(row)
        elif len(segs) == 2:
            rows = {}
            row['title'] = segs[1]
            row['content'] = ""
            if label: row['label'] = label2int[segs[2]]
            data.append(row)
        else:
            data_error.rows.append(line)
    data = pd.DataFrame(data)
    if label:
        data = data[['title', 'content', 'label']]
    else:
        data = data[['title', 'content']]
    return data

def get_train_data():
    result_path = Config.data_path + 'train_data.csv'
    result = get_data(Config.train_file)
    result.fillna("", inplace=True)
    result.to_csv(result_path, index=False, sep='\t')
    return result

