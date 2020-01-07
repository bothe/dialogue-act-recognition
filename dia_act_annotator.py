import os
import time

import numpy as np
import requests
from krippendorff import alpha
from scipy.stats import stats
from sklearn.metrics import classification

from main_swda_elmo_mean import predict_classes_for_elmo_mean
from main_swda_elmo_predictor import predict_classes_for_elmo
from src.final_annotator_utils import convert_predictions_to_indices, ensemble_eda_annotation
from src.utils_float_string import string_to_floats

elmo_feature_retrieval = False
predict_with_elmo = False
predict_with_elmo_mean = False


def predict_da_classes(speaker_id, utterances, utt_id, emotion, link_online=False):
    """ Predicting from text takes 'x' as a list of utterances and
    will require to have ELMo emb server running at port 4004 or online hosting service. """

    if os.path.exists('results/tags.npy'):
        tags = np.load('results/tags.npy')

    if link_online:
        link = "https://d55da20d.eu.ngrok.io/"
    else:
        link = "http://0.0.0.0:4004/"
    x_features = string_to_floats(requests.post(link + 'elmo_embed_words', json={"text": utterances}).json()['result'])

    # Predict with normal elmo features
    swda_elmo_non_con_out, swda_elmo_con_out, swda_elmo_non_con_out_confs, swda_elmo_con_out_confs, \
    swda_elmo_top_con_out, swda_elmo_top_con_out_confs = predict_classes_for_elmo(x_features)

    # Predict with normal elmo mean features
    swda_elmo_features_mean = np.array([item.mean(axis=0) for item in x_features])
    swda_elmo_mean_non_con_out, swda_elmo_mean_con_out, swda_elmo_mean_non_con_out_confs, \
    swda_elmo_mean_con_out_confs = predict_classes_for_elmo_mean(swda_elmo_features_mean)

    # Evaluation of context model predictions
    print('Accuracy comparision between context-based predictions: {}'.format(
        classification.accuracy_score(swda_elmo_con_out, swda_elmo_mean_con_out)))
    print('Kappa (Cohen) score between context-based predictions: {}'.format(
        classification.cohen_kappa_score(swda_elmo_con_out, swda_elmo_mean_con_out)))
    print(classification.classification_report(swda_elmo_con_out, swda_elmo_mean_con_out))
    print('Spearman Correlation between context-based predictions: {}'.format(
        stats.spearmanr(swda_elmo_con_out, swda_elmo_mean_con_out)))
    reliability_data = convert_predictions_to_indices(swda_elmo_con_out, swda_elmo_non_con_out, swda_elmo_mean_con_out,
                                                      swda_elmo_mean_non_con_out, swda_elmo_top_con_out, tags)
    k_alpha = alpha(reliability_data, level_of_measurement='nominal')
    print("Krippendorff's alpha: {}".format(round(k_alpha, 6)))

    from src.relieability_kappa import fleiss_kappa
    fleiss_kappa = fleiss_kappa(reliability_data, 5)
    print("Fleiss' Kappa: {}".format(fleiss_kappa))

    print('Accuracy comparision between context and non-context predictions elmo: {}% elmo_mean: {}% '
          'context-context: {}% non-non-context: {}%'.format(
        classification.accuracy_score(swda_elmo_con_out, swda_elmo_non_con_out),
        classification.accuracy_score(swda_elmo_mean_con_out, swda_elmo_mean_non_con_out),
        classification.accuracy_score(swda_elmo_mean_con_out, swda_elmo_con_out),
        classification.accuracy_score(swda_elmo_mean_non_con_out, swda_elmo_non_con_out)))

    timestamp = time.time()
    # Generate final file of annotations; contains "xx" label for unknown/corrections of EDAs
    rows, assessment = ensemble_eda_annotation(swda_elmo_non_con_out, swda_elmo_mean_non_con_out,
                                               swda_elmo_con_out, swda_elmo_mean_con_out, swda_elmo_top_con_out,
                                               swda_elmo_non_con_out_confs, swda_elmo_mean_non_con_out_confs,
                                               swda_elmo_con_out_confs, swda_elmo_mean_con_out_confs,
                                               swda_elmo_top_con_out_confs,
                                               speaker_id, utterances, utt_id, emotion,
                                               sentiment_labels=[], meld_data=False,
                                               file_name='_final_annotation' + str(timestamp),
                                               write_final_csv=True, write_utterances=True, return_assessment=True)

    print(assessment)
    print('ran swda_dia_act_annotate.py, with total {} number of utterances'.format(len(rows)))
    return rows, assessment, k_alpha, fleiss_kappa