from main_predictor_online import predict_classes
from mocap_annotator import get_mocap_data
from collections import Counter


utterances, emotion, emo_evo, v, a, d = get_mocap_data()

non_con_out, con_out = predict_classes("\r\n".join(utterances[0:10]))

print('debug')
