# from main_predictor_online import predict_classes_from_features
from mocap_annotator import get_mocap_data
from collections import Counter
import numpy as np
from sklearn.metrics import classification
from scipy.stats import stats
from statsmodels.stats.inter_rater import fleiss_kappa
from diswiz.main import predict_das_diswiz

utterances, emotion, emo_evo, v, a, d, speaker_id = get_mocap_data()

con_das, non_con_das, con_da_nums, non_con_da_nums = predict_das_diswiz(utterances)

# iemocap_elmo_features = get_elmo_fea(utterances, mean=False)
# iemocap_elmo_features = np.load('features/iemocap_elmo_features.npy', allow_pickle=True)

# non_con_out, con_out, non_con_out_nums, con_out_nums = predict_classes_from_features(iemocap_elmo_features)
con_das = ['none', 'none'].append(con_das)
print('Accuracy comparision between context- and non-context-based prediction: {}'.format(
    classification.accuracy_score(['none', 'none'].extend(con_das), non_con_das)))

print('Kappa (Cohen) score between context- and non-context-based prediction: {}'.format(
    classification.cohen_kappa_score(con_das, non_con_das)))

print(classification.classification_report(con_das, non_con_das))

print('Spearman Correlation between context- and non-context-based prediction: {}'.format(
    stats.spearmanr(con_das, non_con_das)))

print('debug')
