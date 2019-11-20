from final_annotator_utils import ensemble_annotation
from mocap_data_reader import get_mocap_data
from collections import Counter
import numpy as np
from sklearn.metrics import classification
from scipy.stats import stats

# Get IEMOCAP data
from results.read_predictions_utils import read_all_predictions

utterances, emotion, emo_evo, v, a, d, speaker_id = get_mocap_data()

### Get features
# iemocap_elmo_features = get_elmo_fea(utterances, mean=False)
iemocap_elmo_mean_features = np.load('features/iemocap_elmo_mean_features.npy', allow_pickle=True)
iemocap_elmo_features = np.load('features/iemocap_elmo_features.npy', allow_pickle=True)

## Predict with DISWIZ
# from diswiz.main import predict_das_diswiz
# con_das, non_con_das, con_da_nums, non_con_da_nums = predict_das_diswiz(utterances)

## Predict with normal elmo features
# from main_predictor_online import predict_classes_from_features
# non_con_out, con_out, non_con_out_nums, con_out_nums = predict_classes_from_features(iemocap_elmo_features)

## Predict with elmo mean features
from main_swda_elmo_mean import *

non_con_out, con_out = predict_classes_elmo_mean_features(iemocap_elmo_mean_features)
con_elmo_embs, con_diswiz, non_con_elmo_embs, non_con_diswiz = read_all_predictions()

print('Accuracy comparision between context and non-context predictions elmo: {}% elmo_mean: {}% '
      'context-context: {}% non-non-context: {}%'.format(
    classification.accuracy_score(non_con_elmo_embs, con_elmo_embs),
    classification.accuracy_score(non_con_out, con_out),
    classification.accuracy_score(con_elmo_embs, con_out),
    classification.accuracy_score(non_con_out, non_con_elmo_embs)))

# How to decide which NON_CON_OUT to consider - based on accuracy

print('Kappa (Cohen) score between context- and non-context-based prediction: {}'.format(
    classification.cohen_kappa_score(con_elmo_embs, con_out)))

print(classification.classification_report(con_elmo_embs, con_out))

print('Spearman Correlation between context- and non-context-based prediction: {}'.format(
    stats.spearmanr(con_elmo_embs, con_out)))

np.save('results/elmo_mean_con_out', con_out)
np.save('results/elmo_mean_non_con_out', non_con_out)

rows = ensemble_annotation(non_con_elmo_embs, con_out, con_elmo_embs,
                           speaker_id, utterances, speaker_id,
                           emotion, meld_data=False, file_name='iemocap', write_final_csv=True)

print('debug')
