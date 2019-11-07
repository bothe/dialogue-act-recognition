import numpy as np


def labels_to_indices(labels, classes):
    indices = []
    for item in labels:
        indices.append(classes.index(item))
    return np.array(indices)


def read_all_predictions():
    con_elmo_embs = np.load('results/con_out.npy')
    con_diswiz = np.load('results/con_out_diswiz.npy')
    non_con_elmo_embs = np.load('results/non_con_out.npy')
    non_con_diswiz = np.load('results/non_con_out_diswiz.npy')
    con_elmo_embs[0] = non_con_elmo_embs[0]
    con_elmo_embs[1] = non_con_elmo_embs[1]
    return con_elmo_embs, con_diswiz, non_con_elmo_embs, non_con_diswiz

