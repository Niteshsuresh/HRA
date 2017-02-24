import csv,os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def csv_read_header(filename):
    array = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        array = reader.__next__()
    return array

def csv_read_k_v(filename):
    array = [];
    with open(filename, 'r') as f:
        records = csv.DictReader(f)
        for row in records:
            array.append(row)
    return array

def csv_read(filename, with_header=False):
    array = [];
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            array.append(row)
    return array if with_header else array[1:]

def csv_read_by_colname(filename, colname):
    array = [];
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            array.append(row[colname])
    return filter(None,array)

def combine_maps(maps):
    map_o = {}
    for mp in maps:
        for k,v in mp.items():
            if k not in map_o.keys():
                map_o[k] = [v]
            else:
                map_o[k].append(v)
    return map_o

def prepare_for_plot(headers, rows):
    headers = list(filter(lambda x: x != 'left', headers))
    v = []
    for header in headers:
        v = v+list(map(lambda x: {header: x[header]}, rows))
    map_o = combine_maps(v)
    for  k , v in map_o.items():
        data = dict(Counter(v).most_common(30))
        x = np.arange(len(data))
        y = data.values()
        plt.figure(figsize=(10,10))
        plt.bar(x, y)
        plt.tight_layout()
        plt.tight_layout()
        plt.xticks(x + 0.5, data.keys(), rotation='vertical')
        plt.savefig(k)
        plt.show()


def process():
    cols = csv_read_k_v('data/HR_comma_sep.csv')
    headers = csv_read_header('data/HR_comma_sep.csv')
    left = list(filter(lambda o: filter_left(o), cols))
    prepare_for_plot(headers, left)

def left_emplys():
    in_data = csv_read('data/HR_comma_sep.csv')
    sal_map = {'low':1, 'medium':2, 'high':3}
    indus_map = {'technical':1,'sales':2, 'accounting':3, 'hr': 4,'support':5, 'management':6, 'IT':7, 'product_mng':8, 'marketing':9, 'RandD':10 }

    out_set = []
    train_set = []
    for data in in_data:
        out_set.append(float(data[-4]))
        data[-1] = float(sal_map[data[-1]])
        data[-2] = float(indus_map[data[-2]])
        a = data[0:5] + data[7:]
        train_set.append(to_float(a))
    return {'data': train_set, 'target': out_set}

def to_float(array):
    a = np.array(array, dtype='float')
    return a

def trainer(data):
    model = GaussianNB()
    model.fit(data['data'],data['target'])
    return model


def train():
    empls = left_emplys()
    data = empls['data']
    empls['data'] = data[0: int(len(data)/2)]
    empls['target'] = empls['target'][0: int(len(empls['target'])/2)]
    classifier = trainer(empls)
    return classifier

def accuracy():
    trainer = train()
    train_set = left_emplys()
    expected = train_set['target'][int(len(train_set['target'])/2):]
    train_set = train_set['data'][int(len(train_set['data'])/2):]
    predicted = trainer.predict(train_set)
    print(metrics.classification_report(expected, predicted))

accuracy()


def filter_left(row):
    if int(row['left']) == 1:
        del row['left']
        return True
    return False
