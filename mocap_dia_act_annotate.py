# from main_predictor_online import predict_classes
from mocap_annotator import get_mocap_data
from collections import Counter
from elmo_features import get_elmo_fea
import numpy as np

utterances, emotion, emo_evo, v, a, d = get_mocap_data()
iemocap_elmo_features = get_elmo_fea(utterances)

# non_con_out, con_out = predict_classes("\r\n".join(utterances), link_online=False)
np.save('features/iemocap_elmo_features.npy', iemocap_elmo_features)


print('debug')
