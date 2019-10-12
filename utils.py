import csv
import numpy as np


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
    for i in range(0, len(x) - seq_length, 1):
        seq_in = x[i:i + seq_length]
        seq_out = y[i + seq_length - 1]
        dataX.append([seq_in])
        dataY.append(seq_out)
    return (np.array(dataX).squeeze(axis=1)), np.array(dataY)


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


def returnlist(filename):
    fo = open(filename, "r")
    lines = fo.readlines()
    fo.close()
    return lines


def padSequences(x, toPadding):
    new_x = []
    for item in x:
        if len(item) > 20:
            new_x.append(item[0:20])
        elif len(item) < 20:
            # print(item.shape)
            for i in range(20):
                item = np.append([item], toPadding).reshape(len(item) + 1, 1024)
                if len(item) == 20:
                    new_x.append(item)
        elif len(item) == 20:
            new_x.append(item[0:20])

    return np.array(new_x)
