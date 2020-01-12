import os
import time

import numpy as np
import requests
from flask import Flask, jsonify, request
from krippendorff import alpha
from scipy.stats import stats
from sklearn.metrics import classification

from src.final_annotator_utils import convert_predictions_to_indices, ensemble_eda_annotation
from src.utils_float_string import string_to_floats, str_utils

app = Flask(__name__)


@app.route("/predict_das", methods=['GET', 'POST'])
def index():
    """ Predicting from text takes 'x' as a list of utterances and
    will require to have ELMo emb server running at port 4004 or online hosting service. """
    value = request.json['text']
    speaker_id, utterances, utt_id, emotion = str_utils(text=value, mode='decode')
    if os.path.exists('results/tags.npy'):
        tags = np.load('results/tags.npy')
    link_online = False
    if link_online:
        link = "https://d55da20d.eu.ngrok.io/"
    else:
        link = "http://0.0.0.0:4004/"
    utterances_post = '\r\n'.join(utterances)
    x_features = string_to_floats(requests.post(link + "elmo_embed_words",
                                                json={"text": utterances_post}).json()["result"])

    # Predict with normal elmo features
    from main_swda_elmo_predictor import predict_classes_for_elmo
    swda_elmo_non_con_out, swda_elmo_con_out, swda_elmo_non_con_out_confs, swda_elmo_con_out_confs, \
    swda_elmo_top_con_out, swda_elmo_top_con_out_confs = predict_classes_for_elmo(x_features)

    # Predict with normal elmo mean features
    from main_swda_elmo_mean import predict_classes_for_elmo_mean
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

    try:
        from src.relieability_kappa import fleiss_kappa
        fleiss_kappa_score = fleiss_kappa(reliability_data, 5)
        print("Fleiss' Kappa: {}".format(fleiss_kappa_score))
    except IndexError:
        print("Could not compute Fleiss' Kappa score, due to insufficient categories in the final annotations!")
        fleiss_kappa_score = None

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
                                               file_name='annotations_' + str(timestamp),
                                               write_final_csv=True, write_utterances=True, return_assessment=True)

    print('ran swda_dia_act_annotate.py, with total {} number of utterances'.format(len(rows)))
    # return rows, assessment, k_alpha, fleiss_kappa_score
    # return jsonify({'result': rows, 'assessment': assessment, 'k_alpha': k_alpha,
    #                'fleiss_kappa_score': fleiss_kappa_score})
    res = str_utils(speaker_id=speaker_id, utterances=utterances, utt_id=utt_id, emotion=emotion, mode='encode')
    #    return jsonify({'result': res})

    result_text = []
    for item in rows:
        result_text.append(item['eda1'] + '$$' + item['eda2'] + '$$' + item['eda3'] + '$$' +
                           item['eda4'] + '$$' + item['eda5'] + '$$' + item['EDA'] + '$$' +
                           item['all_match'] + '$$' + item['con_match'] + '$$' + item['match'])

    result_text = '???'.join(result_text)
    return jsonify({'result': result_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4004)
