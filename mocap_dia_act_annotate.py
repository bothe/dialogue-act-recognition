import os

import numpy as np
from krippendorff import alpha
from scipy.stats import stats
from sklearn.metrics import classification

from src.final_annotator_utils import convert_predictions_to_indices, ensemble_eda_annotation
from src.relieability_kappa import fleiss_kappa
from src.mocap_data_reader import get_mocap_data

elmo_feature_retrieval = False
predict_with_elmo = False
predict_with_elmo_mean = False

if os.path.exists('results/tags.npy'):
    tags = np.load('results/tags.npy')

utterances, emotion, emo_evo, v, a, d, speaker_id, utt_id = get_mocap_data(read_from_csv=True, write=True)

if elmo_feature_retrieval:
    from elmo_features import get_elmo_embs

    iemocap_elmo_features = get_elmo_embs(utterances, mean=False)
    np.save('features/iemocap_elmo_features', iemocap_elmo_features)
elif predict_with_elmo or predict_with_elmo_mean:
    iemocap_elmo_features = np.load('features/iemocap_elmo_features.npy', allow_pickle=True)
    # iemocap_elmo_mean_features = np.load('features/iemocap_elmo_mean_features.npy', allow_pickle=True)

# Predict with normal elmo features
if predict_with_elmo:
    from main_swda_elmo_predictor import predict_classes_for_elmo

    mocap_elmo_non_con_out, mocap_elmo_con_out, mocap_elmo_non_con_out_confs, mocap_elmo_con_out_confs, \
    mocap_elmo_top_con_out, mocap_elmo_top_con_out_confs = predict_classes_for_elmo(iemocap_elmo_features)
    np.save('model_output_labels/mocap_elmo_con_out', mocap_elmo_con_out)
    np.save('model_output_labels/mocap_elmo_non_con_out', mocap_elmo_non_con_out)
    np.save('model_output_labels/mocap_elmo_con_out_confs', mocap_elmo_con_out_confs)
    np.save('model_output_labels/mocap_elmo_non_con_out_confs', mocap_elmo_non_con_out_confs)
    np.save('model_output_labels/mocap_elmo_top_con_out', mocap_elmo_top_con_out)
    np.save('model_output_labels/mocap_elmo_top_con_out_confs', mocap_elmo_top_con_out_confs)
else:
    mocap_elmo_con_out = np.load('model_output_labels/mocap_elmo_con_out.npy')
    mocap_elmo_non_con_out = np.load('model_output_labels/mocap_elmo_non_con_out.npy')
    mocap_elmo_con_out_confs = np.load('model_output_labels/mocap_elmo_con_out_confs.npy')
    mocap_elmo_non_con_out_confs = np.load('model_output_labels/mocap_elmo_non_con_out_confs.npy')
    mocap_elmo_top_con_out = np.load('model_output_labels/mocap_elmo_top_con_out.npy')
    mocap_elmo_top_con_out_confs = np.load('model_output_labels/mocap_elmo_top_con_out_confs.npy')

# Predict with normal elmo mean features
if predict_with_elmo_mean:
    from main_swda_elmo_mean import predict_classes_for_elmo_mean

    iemocap_elmo_features_mean = np.array([item.mean(axis=0) for item in iemocap_elmo_features])
    mocap_elmo_mean_non_con_out, mocap_elmo_mean_con_out, mocap_elmo_mean_non_con_out_confs, \
    mocap_elmo_mean_con_out_confs = predict_classes_for_elmo_mean(iemocap_elmo_features_mean)

    np.save('model_output_labels/mocap_elmo_mean_con_out', mocap_elmo_mean_con_out)
    np.save('model_output_labels/mocap_elmo_mean_non_con_out', mocap_elmo_mean_non_con_out)
    np.save('model_output_labels/mocap_elmo_mean_con_out_confs', mocap_elmo_mean_con_out_confs)
    np.save('model_output_labels/mocap_elmo_mean_non_con_out_confs', mocap_elmo_mean_non_con_out_confs)
else:
    mocap_elmo_mean_con_out = np.load('model_output_labels/mocap_elmo_mean_con_out.npy')
    mocap_elmo_mean_non_con_out = np.load('model_output_labels/mocap_elmo_mean_non_con_out.npy')
    mocap_elmo_mean_con_out_confs = np.load('model_output_labels/mocap_elmo_mean_con_out_confs.npy')
    mocap_elmo_mean_non_con_out_confs = np.load('model_output_labels/mocap_elmo_mean_non_con_out_confs.npy')

# Evaluation of context model predictions
print('Accuracy comparision between context-based predictions: {}'.format(
    classification.accuracy_score(mocap_elmo_con_out, mocap_elmo_mean_con_out)))
print('Kappa (Cohen) score between context-based predictions: {}'.format(
    classification.cohen_kappa_score(mocap_elmo_con_out, mocap_elmo_mean_con_out)))
print(classification.classification_report(mocap_elmo_con_out, mocap_elmo_mean_con_out))
print('Spearman Correlation between context-based predictions: {}'.format(
    stats.spearmanr(mocap_elmo_con_out, mocap_elmo_mean_con_out)))
reliability_data = convert_predictions_to_indices(mocap_elmo_con_out, mocap_elmo_non_con_out, mocap_elmo_mean_con_out,
                                                  mocap_elmo_mean_non_con_out, mocap_elmo_top_con_out, tags)
k_alpha = alpha(reliability_data, level_of_measurement='nominal')
print("Krippendorff's alpha: {}".format(round(k_alpha, 6)))

fleiss_kappa_score = fleiss_kappa(reliability_data, 5)

print('Accuracy comparision between context and non-context predictions elmo: {}% elmo_mean: {}% '
      'context-context: {}% non-non-context: {}%'.format(
    classification.accuracy_score(mocap_elmo_con_out, mocap_elmo_non_con_out),
    classification.accuracy_score(mocap_elmo_mean_con_out, mocap_elmo_mean_non_con_out),
    classification.accuracy_score(mocap_elmo_mean_con_out, mocap_elmo_con_out),
    classification.accuracy_score(mocap_elmo_mean_non_con_out, mocap_elmo_non_con_out)))

# Generate final file of annotations; contains "xx" label for unknown/corrections of EDAs
row = ensemble_eda_annotation(mocap_elmo_non_con_out, mocap_elmo_mean_non_con_out,
                              mocap_elmo_con_out, mocap_elmo_mean_con_out, mocap_elmo_top_con_out,
                              mocap_elmo_non_con_out_confs, mocap_elmo_mean_non_con_out_confs,
                              mocap_elmo_con_out_confs, mocap_elmo_mean_con_out_confs, mocap_elmo_top_con_out_confs,
                              utt_id, utterances, speaker_id, emotion,
                              sentiment_labels=[], meld_data=False,
                              file_name='iemocap_no_utts', write_final_csv=False, write_utterances=False)

print('ran mocap_dia_act_annotate.py, with total {} number of utterances'.format(len(utterances)))
