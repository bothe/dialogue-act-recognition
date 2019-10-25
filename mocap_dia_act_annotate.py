from main_predictor_online import predict_classes_from_features
from mocap_annotator import get_mocap_data
from collections import Counter
import numpy as np

x = np.load("features/iemocap_elmo_features.npy")

utterances, emotion, emo_evo, v, a, d = get_mocap_data()

non_con_out, con_out = predict_classes_from_features(x)

print('debug')
