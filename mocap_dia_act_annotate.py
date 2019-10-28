from main_predictor_online import predict_classes_from_features
from mocap_annotator import get_mocap_data
from collections import Counter
from elmo_features import get_elmo_fea
import numpy as np
from sklearn.metrics import classification
from scipy.stats import stats
from statsmodels.stats.inter_rater import fleiss_kappa
from diswiz.main import predict_das_diswiz

utterances, emotion, emo_evo, v, a, d = get_mocap_data()

xxx = predict_das_diswiz(utterances)

# iemocap_elmo_features = get_elmo_fea(utterances, mean=False)
iemocap_elmo_features = np.load('features/iemocap_elmo_features.npy', allow_pickle=True)

non_con_out, con_out, non_con_out_nums, con_out_nums = predict_classes_from_features(iemocap_elmo_features)

print('Accuracy comparision between context- and non-context-based prediction: {}'.format(
    classification.accuracy_score(con_out_nums, non_con_out_nums)))

print('Kappa (Cohen) score between context- and non-context-based prediction: {}'.format(
    classification.cohen_kappa_score(con_out_nums, non_con_out_nums)))

print(classification.classification_report(con_out_nums, non_con_out_nums))

print('Spearman Correlation between context- and non-context-based prediction: {}'.format(
    stats.spearmanr(con_out_nums, non_con_out_nums)))


np.save("results/con_out", con_out)
np.save("results/con_out_nums", con_out_nums)
np.save("results/non_con_out", non_con_out)
np.save("results/non_con_out_nums", non_con_out_nums)

print('debug')
