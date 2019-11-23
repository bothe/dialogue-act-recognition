import os

import numpy as np
from krippendorff import alpha
from scipy.stats import stats
from sklearn.metrics import classification

from final_annotator_utils import ensemble_annotation, convert_predictions_to_indices
from mocap_data_reader import get_mocap_data
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
from main_predictor_online import predict_classes_for_elmo
non_con_out, con_out, non_con_out_nums, con_out_nums = predict_classes_for_elmo(iemocap_elmo_features)

con_elmo_embs, con_diswiz, non_con_elmo_embs, non_con_diswiz = read_all_predictions()

tags_file_name = 'results/tags.npy'
elmo_mean_con_out_file_name = 'results/elmo_mean_con_out.npy'
elmo_mean_non_con_out_file_name = 'results/elmo_mean_non_con_out.npy'

## Predict with elmo mean features
if os.path.exists(elmo_mean_con_out_file_name) and os.path.exists(elmo_mean_non_con_out_file_name):
    elmo_mean_non_con_out, elmo_mean_con_out = np.load(elmo_mean_non_con_out_file_name), np.load(
        elmo_mean_con_out_file_name)
    tags = np.load(tags_file_name)
else:
    from main_swda_elmo_mean import *

    elmo_mean_non_con_out, elmo_mean_con_out = predict_classes_elmo_mean_features(iemocap_elmo_mean_features)
    np.save(elmo_mean_con_out_file_name, elmo_mean_con_out)
    np.save(elmo_mean_non_con_out_file_name, elmo_mean_non_con_out)
    np.save(tags_file_name, tags)

reliability_data = convert_predictions_to_indices(elmo_mean_con_out, elmo_mean_non_con_out, con_elmo_embs,
                                                  non_con_elmo_embs, tags)
k_alpha = alpha(reliability_data, level_of_measurement='nominal')
print("Krippendorff's alpha: {}".format(round(k_alpha, 6)))

print('Accuracy comparision between context and non-context predictions elmo: {}% elmo_mean: {}% '
      'context-context: {}% non-non-context: {}%'.format(
    classification.accuracy_score(non_con_elmo_embs, con_elmo_embs),
    classification.accuracy_score(elmo_mean_non_con_out, elmo_mean_con_out),
    classification.accuracy_score(con_elmo_embs, elmo_mean_con_out),
    classification.accuracy_score(elmo_mean_non_con_out, non_con_elmo_embs)))

# How to decide which NON_CON_OUT to consider - based on accuracy

print('Kappa (Cohen) score between context-context-based predictions: {}'.format(
    classification.cohen_kappa_score(con_elmo_embs, elmo_mean_con_out)))

print(classification.classification_report(con_elmo_embs, elmo_mean_con_out))

print('Spearman Correlation between context-based predictions: {}'.format(
    stats.spearmanr(con_elmo_embs, elmo_mean_con_out)))

rows = ensemble_annotation(non_con_elmo_embs, elmo_mean_con_out, con_elmo_embs,
                           speaker_id, utterances, speaker_id,
                           emotion, meld_data=False, file_name='iemocap', write_final_csv=False)

print('ran mocap_dia_act_annotate.py')
