import numpy as np
from sklearn.metrics import classification
from collections import Counter

con_elmo_embs = np.load('results/con_out.npy')
con_diswiz = np.load('results/con_out_diswiz.npy')
# con_diswiz = ['None', 'None'].extend(con_diswiz)
non_con_elmo_embs = np.load('results/non_con_out.npy')
non_con_diswiz = np.load('results/non_con_out_diswiz.npy')

all_classes = Counter(list(con_diswiz) + list(con_elmo_embs) + list(non_con_diswiz) + list(non_con_elmo_embs))
classes = list(all_classes.keys())


def labels_to_indices(labels, classes):
    indices = []
    for item in labels:
        indices.append(classes.index(item))
    return np.array(indices)


con_elmo_embs = labels_to_indices(con_elmo_embs, classes)
con_diswiz = labels_to_indices(con_diswiz, classes)
non_con_elmo_embs = labels_to_indices(non_con_elmo_embs, classes)
non_con_diswiz = labels_to_indices(non_con_diswiz, classes)

print('debug')
