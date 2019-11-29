import os

import numpy as np
from krippendorff import alpha
from scipy.stats import stats
from sklearn.metrics import classification

from MELD.utils.read_meld import *
from src.final_annotator_utils import convert_predictions_to_indices, ensemble_eda_annotation
from src.relieability_kappa import fleissKappa

elmo_feature_retrieval = False
predict_with_elmo = False
predict_with_elmo_mean = False
predict_with_diswiz = False

if os.path.exists('results/tags.npy'):
    tags = np.load('results/tags.npy')

if elmo_feature_retrieval:
    from elmo_features import get_elmo_embs

    meld_elmo_features_test = get_elmo_embs(utt_test_data, mean=False)
    np.save('features/meld_elmo_features_test', meld_elmo_features_test)
    meld_elmo_features_dev = get_elmo_embs(utt_dev_data, mean=False)
    np.save('features/meld_elmo_features_dev', meld_elmo_features_dev)
    meld_elmo_features_train = get_elmo_embs(utt_train_data, mean=False)
    np.save('features/meld_elmo_features_train', meld_elmo_features_train)
elif predict_with_elmo or predict_with_elmo_mean:
    meld_elmo_features_test = np.load('features/meld_elmo_features_test.npy', allow_pickle=True)
    meld_elmo_features_dev = np.load('features/meld_elmo_features_dev.npy', allow_pickle=True)
    meld_elmo_features_train = np.load('features/meld_elmo_features_train.npy', allow_pickle=True)

# Predict with normal elmo features
if predict_with_elmo:
    from main_swda_elmo_predictor import predict_classes_for_elmo

    concatenated_vectors = np.concatenate((meld_elmo_features_train, meld_elmo_features_dev, meld_elmo_features_test))
    meld_elmo_non_con_out, meld_elmo_con_out, meld_elmo_non_con_out_confs, meld_elmo_con_out_confs, \
    meld_elmo_top_con_out, meld_elmo_top_con_out_confs = predict_classes_for_elmo(concatenated_vectors)

    np.save('model_output_labels/meld_elmo_con_out', meld_elmo_con_out)
    np.save('model_output_labels/meld_elmo_non_con_out', meld_elmo_non_con_out)
    np.save('model_output_labels/meld_elmo_con_out_confs', meld_elmo_con_out_confs)
    np.save('model_output_labels/meld_elmo_non_con_out_confs', meld_elmo_non_con_out_confs)
    np.save('model_output_labels/meld_elmo_top_con_out', meld_elmo_top_con_out)
    np.save('model_output_labels/meld_elmo_top_con_out_confs', meld_elmo_top_con_out_confs)
else:
    meld_elmo_con_out = np.load('model_output_labels/meld_elmo_con_out.npy')
    meld_elmo_non_con_out = np.load('model_output_labels/meld_elmo_non_con_out.npy')
    meld_elmo_con_out_confs = np.load('model_output_labels/meld_elmo_con_out_confs.npy')
    meld_elmo_non_con_out_confs = np.load('model_output_labels/meld_elmo_non_con_out_confs.npy')
    meld_elmo_top_con_out = np.load('model_output_labels/meld_elmo_top_con_out.npy')
    meld_elmo_top_con_out_confs = np.load('model_output_labels/meld_elmo_top_con_out_confs.npy')

if predict_with_elmo_mean:
    from main_swda_elmo_mean import *

    meld_elmo_features_test_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_test])
    meld_elmo_features_dev_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_dev])
    meld_elmo_features_train_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_train])
    concatenated_mean_vectors = np.concatenate((meld_elmo_features_train_mean, meld_elmo_features_dev_mean,
                                                meld_elmo_features_test_mean))
    meld_elmo_mean_non_con_out, meld_elmo_mean_con_out, meld_elmo_mean_non_con_out_confs, \
    meld_elmo_mean_con_out_confs = predict_classes_for_elmo_mean(concatenated_mean_vectors)

    np.save('model_output_labels/meld_elmo_mean_con_out', meld_elmo_mean_con_out)
    np.save('model_output_labels/meld_elmo_mean_non_con_out', meld_elmo_mean_non_con_out)
    np.save('model_output_labels/meld_elmo_mean_con_out_confs', meld_elmo_mean_con_out_confs)
    np.save('model_output_labels/meld_elmo_mean_non_con_out_confs', meld_elmo_mean_non_con_out_confs)
else:
    meld_elmo_mean_con_out = np.load('model_output_labels/meld_elmo_mean_con_out.npy')
    meld_elmo_mean_non_con_out = np.load('model_output_labels/meld_elmo_mean_non_con_out.npy')
    meld_elmo_mean_con_out_confs = np.load('model_output_labels/meld_elmo_mean_con_out_confs.npy')
    meld_elmo_mean_non_con_out_confs = np.load('model_output_labels/meld_elmo_mean_non_con_out_confs.npy')

utt_Speaker = utt_Speaker_train + utt_Speaker_dev + utt_Speaker_test
utt_data = utt_train_data + utt_dev_data + utt_test_data
utt_id_data = utt_id_train_data + utt_id_dev_data + utt_id_test_data
utt_Emotion_data = utt_Emotion_train_data + utt_Emotion_dev_data + utt_Emotion_test_data
utt_Sentiment_data = utt_Sentiment_train_data + utt_Sentiment_dev_data + utt_Sentiment_test_data

if predict_with_diswiz:
    from diswiz.main import predict_das_diswiz

    meld_diswiz_con_out, meld_diswiz_non_con_out, meld_diswiz_con_out_confs, \
    meld_diswiz_non_con_out_confs = predict_das_diswiz(utt_data)
    np.save('model_output_labels/meld_diswiz_con_out', meld_diswiz_con_out)
    np.save('model_output_labels/meld_diswiz_con_out_confs', meld_diswiz_con_out_confs)
else:
    meld_diswiz_con_out = np.load('model_output_labels/meld_diswiz_con_out.npy')
    meld_diswiz_con_out_confs = np.load('model_output_labels/meld_diswiz_con_out_confs.npy')

# Evaluation of context model predictions
print('Accuracy comparision between context-based predictions: {}'.format(
    classification.accuracy_score(meld_elmo_con_out, meld_elmo_mean_con_out)))
print('Kappa (Cohen) score between context-based predictions: {}'.format(
    classification.cohen_kappa_score(meld_elmo_con_out, meld_elmo_mean_con_out)))
print(classification.classification_report(meld_elmo_con_out, meld_elmo_mean_con_out))
print('Spearman Correlation between context-based predictions: {}'.format(
    stats.spearmanr(meld_elmo_con_out, meld_elmo_mean_con_out)))
reliability_data = convert_predictions_to_indices(meld_elmo_con_out, meld_elmo_non_con_out, meld_elmo_mean_con_out,
                                                  meld_elmo_mean_non_con_out, meld_elmo_top_con_out, tags)
k_alpha = alpha(reliability_data, level_of_measurement='nominal')
print("Krippendorff's alpha: {}".format(round(k_alpha, 6)))

fleiss_kappa_score = fleissKappa(reliability_data, 5)

# Generate final file of annotations; contains "xx" label for unknown/corrections of EDAs
row = ensemble_eda_annotation(meld_elmo_non_con_out, meld_elmo_mean_non_con_out,
                              meld_elmo_con_out, meld_elmo_mean_con_out, meld_elmo_top_con_out,
                              meld_elmo_non_con_out_confs, meld_elmo_mean_non_con_out_confs,
                              meld_elmo_con_out_confs, meld_elmo_mean_con_out_confs, meld_elmo_top_con_out_confs,
                              utt_Speaker, utt_data, utt_id_data, utt_Emotion_data,
                              sentiment_labels=utt_Sentiment_data, meld_data=True,
                              file_name='meld_emotion', write_final_csv=True)

print('ran meld_dia_act_annotate.py, with total {} number of utterances'.format(len(utt_data)))
