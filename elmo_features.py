from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers import word_tokenizer
import numpy as np
import time, sys
import pyprind

elmo = ElmoEmbedder()


def get_elmo_fea(data, mean=True, print_length=False):
    n = len(data)
    tk = word_tokenizer.WordTokenizer()
    tokens = tk.batch_tokenize(data)
    idx = []
    bar = pyprind.ProgBar(n, stream=sys.stdout)
    for i in range(len(tokens)):
        idx.append([x.idx for x in tokens[i]])
        tokens[i] = [x.text for x in tokens[i]]
        bar.update()
    bar = pyprind.ProgBar(n, stream=sys.stdout)
    vectors = []
    for seq in tokens:
        vector = elmo.embed_sentence(seq[0:21])
        if mean:
            vectors.append(np.mean(vector[2], axis=0))
        else:
            vectors.append(vector[2])
        if print_length:
            print('Length of a sequence: {} with final emb vector shape: {}'.format(len(seq), vector.shape))
        bar.update()
    return vectors


def get_elmo_tokens(data):
    n = len(data)
    # elmo = ElmoEmbedder()
    tk = word_tokenizer.WordTokenizer()
    tokens = tk.batch_tokenize(data)
    idx = []
    bar = pyprind.ProgBar(n, stream=sys.stdout)
    for i in range(len(tokens)):
        idx.append([x.idx for x in tokens[i]])
        tokens[i] = [x.text for x in tokens[i]]
        bar.update()
    return tokens


def get_elmo_features(data):
    def get_nearest(slot, target):
        for i in range(target, -1, -1):
            if i in slot:
                return i

    tk = word_tokenizer.WordTokenizer()
    tokens = tk.batch_tokenize(data)
    idx = []

    for i in range(len(tokens)):
        idx.append([x.idx for x in tokens[i]])
        tokens[i] = [x.text for x in tokens[i]]

    vectors = elmo.embed_sentences(tokens)

    ans = []
    for i, vector in enumerate([v for v in vectors]):
        P_l = data.iloc[i].Pronoun
        A_l = data.iloc[i].A.split()
        B_l = data.iloc[i].B.split()

        P_offset = data.iloc[i]['Pronoun-offset']
        A_offset = data.iloc[i]['A-offset']
        B_offset = data.iloc[i]['B-offset']

        if P_offset not in idx[i]:
            P_offset = get_nearest(idx[i], P_offset)
        if A_offset not in idx[i]:
            A_offset = get_nearest(idx[i], A_offset)
        if B_offset not in idx[i]:
            B_offset = get_nearest(idx[i], B_offset)

        emb_P = np.mean(vector[1:3, idx[i].index(P_offset), :], axis=0, keepdims=True)

        emb_A = np.mean(vector[1:3, idx[i].index(A_offset):idx[i].index(A_offset) + len(A_l), :], axis=(1, 0),
                        keepdims=True)
        emb_A = np.squeeze(emb_A, axis=0)

        emb_B = np.mean(vector[1:3, idx[i].index(B_offset):idx[i].index(B_offset) + len(B_l), :], axis=(1, 0),
                        keepdims=True)
        emb_B = np.squeeze(emb_B, axis=0)

        ans.append(np.concatenate([emb_A, emb_B, emb_P], axis=1))

    emb = np.concatenate(ans, axis=0)
    return emb
