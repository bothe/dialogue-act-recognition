# from main_predictor_online import predict_classes_from_features
from elmo_features import get_elmo_fea
from main_predictor_online import predict_classes_from_features
from collections import Counter
import numpy as np
from sklearn.metrics import classification
from scipy.stats import stats
from MELD.utils.read_meld import *

elmo_feature_retrieval = True
if elmo_feature_retrieval:
    # meld_elmo_features_test = get_elmo_fea(utt_test_data, mean=False)
    # np.save('features/meld_elmo_features_test', meld_elmo_features_test)

    meld_elmo_features_dev = get_elmo_fea(utt_dev_data, mean=False)
    np.save('features/meld_elmo_features_dev', meld_elmo_features_dev)

    meld_elmo_features_train = get_elmo_fea(utt_train_data, mean=False)
    np.save('features/meld_elmo_features_train', meld_elmo_features_train)
else:
    meld_elmo_features_test = np.save('features/meld_elmo_features_test.npy', allow_pickle=True)
    meld_elmo_features_dev = np.save('features/meld_elmo_features_dev.npy', allow_pickle=True)
    meld_elmo_features_train = np.save('features/meld_elmo_features_train.npy', allow_pickle=True)

non_con_out, con_out, non_con_out_nums, con_out_nums = predict_classes_from_features(meld_elmo_features_train)

con_out = ['none', 'none'].append(con_out)
print('Accuracy comparision between context- and non-context-based prediction: {}'.format(
    classification.accuracy_score(['none', 'none'].extend(con_out), non_con_out)))

print('Kappa (Cohen) score between context- and non-context-based prediction: {}'.format(
    classification.cohen_kappa_score(con_out, non_con_out)))

print(classification.classification_report(con_out, non_con_out))

print('Spearman Correlation between context- and non-context-based prediction: {}'.format(
    stats.spearmanr(con_out, non_con_out)))

print('debug')
