import numpy as np
from collections import Counter
from final_annotator_utils import *
from mocap_data_reader import get_mocap_data
from sklearn.metrics import classification
from scipy.stats import stats

con_out = np.load('results/con_out.npy')
con_out_mean = np.load('results/con_out.npy')
non_con_out = np.load('results/non_con_out.npy')
con_out[0] = non_con_out[0]
con_out[1] = non_con_out[1]

# Evaluation of context model predictions
print('Accuracy comparision between context- and non-context-based prediction: {}'.format(
    classification.accuracy_score(con_out, con_out_mean)))
print('Kappa (Cohen) score between context- and non-context-based prediction: {}'.format(
    classification.cohen_kappa_score(con_out, con_out_mean)))
print(classification.classification_report(con_out, con_out_mean))
print('Spearman Correlation between context- and non-context-based prediction: {}'.format(
    stats.spearmanr(con_out, con_out_mean)))

utterances, emotion, emo_evo, v, a, d, speaker_id = get_mocap_data()

rows = ensemble_annotation(non_con_out, con_out, con_out, speaker_id, utterances,
                           speaker_id, emotion, sentiment_labels=[],
                           meld_data=False, file_name='mocap_emotion', write_final_csv=True)
print('debug')
