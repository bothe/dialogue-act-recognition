import csv, re
import numpy as np
from collections import Counter


def read_files(filePath):
    # READ FILE
    X, Y, Z, S = [], [], [], []

    with open(filePath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['DamslActTag'] != '+':
                S.append(row['CallerID'])
                X.append(row['Text'])
                Y.append(row['ActTag'])
                Z.append(row['DamslActTag'])
    return S, X, Y, Z


def preparedata(x, y, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(x) - seq_length + 1, 1):
        seq_in = x[i:i + seq_length]
        seq_out = y[i + seq_length - 1]
        dataX.append([seq_in])
        dataY.append(seq_out)
    return (np.array(dataX).squeeze(axis=1)), np.array(dataY)


def prepare_input_data(x, seq_length):
    print(len(x))
    dataX = []
    for i in range(0, len(x) - seq_length + 1, 1):
        seq_in = x[i:i + seq_length]
        dataX.append([seq_in])
    print(len(dataX))
    return (np.array(dataX).squeeze(axis=1))


def cat_3classes(y_avg):
    y_train = []
    threshold = 0.05
    for item in y_avg:
        if item >= threshold:
            y_train.append(0)
        elif -threshold < item < threshold:
            y_train.append(1)
        elif item <= -threshold:
            y_train.append(2)
    return y_train


def prepare_targets(Ztrain, Ztest):
    tag, num = [], []
    for i, j in enumerate((Counter(Ztrain)).keys()):
        tag.append(j)
        num.append(i)
    Y_train = []
    for i in Ztrain:
        Y_train.append(tag.index(i))
    Y_test = []
    for i in Ztest:
        Y_test.append(tag.index(i))

    tag[0] = 'fo_o'
    return tag, num, Y_test, Y_train


def returnlist(filename):
    fo = open(filename, "r")
    lines = fo.readlines()
    fo.close()
    return lines


def readMRDA(file):
    X, Y, Y1, Z = [], [], [], []
    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.split(',')[3] == '':
                pass
            else:
                X.append(line.split(',')[1])
                Y.append(line.split(',')[3])
                Y1.append(re.split(r"[|^.:]+", line.split(',')[3])[0].replace('-', '').replace('--', ''))

    return X, Y, Y1

# file = '/informatik2/wtm/home/bothe/crb/aswda/MRDA_Aggregated.out'
# X, Y, Y1 = readMRDA(file)
# print('debug')
