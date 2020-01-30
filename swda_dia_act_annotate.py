import os

import numpy as np
from keras.utils import to_categorical
from krippendorff import alpha
from scipy.stats import stats
from sklearn.metrics import classification

from src.final_annotator_utils import convert_predictions_to_indices, ensemble_eda_annotation
from src.relieability_kappa import fleiss_kappa
from src.utils import read_files, categorize_raw_data

elmo_feature_retrieval = False
predict_with_elmo = False
predict_with_elmo_mean = False

if os.path.exists('results/tags.npy'):
    tags = np.load('results/tags.npy')

# utterances, emotion, emo_evo, v, a, d, speaker_id, utt_id = get_swda_data(read_from_csv=True, write=True)
trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
toPadding = np.load('features/pad_a_token.npy', allow_pickle=True)
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
target_category_test = to_categorical(Y_test, len(tags))
target_category_train = to_categorical(Y_train, len(tags))
utterances = Xtest
speaker_id = SidTest
emotion = speaker_id

if elmo_feature_retrieval:
    from elmo_features import get_elmo_embs

    swda_elmo_features = get_elmo_embs(utterances, mean=False)
    np.save('features/ieswda_elmo_features', swda_elmo_features)
elif predict_with_elmo or predict_with_elmo_mean:
    swda_elmo_features = np.load('features/X_test_elmo_features.npy', allow_pickle=True)
    X_test_mean = np.load('features/X_test_elmo_mean_features.npy', allow_pickle=True)

# Predict with normal elmo features
if predict_with_elmo:
    from main_swda_elmo_predictor import predict_classes_for_elmo

    swda_elmo_non_con_out, swda_elmo_con_out, swda_elmo_non_con_out_confs, swda_elmo_con_out_confs, \
    swda_elmo_top_con_out, swda_elmo_top_con_out_confs = predict_classes_for_elmo(swda_elmo_features)
    np.save('model_output_swda_labels/swda_elmo_con_out', swda_elmo_con_out)
    np.save('model_output_swda_labels/swda_elmo_non_con_out', swda_elmo_non_con_out)
    np.save('model_output_swda_labels/swda_elmo_con_out_confs', swda_elmo_con_out_confs)
    np.save('model_output_swda_labels/swda_elmo_non_con_out_confs', swda_elmo_non_con_out_confs)
    np.save('model_output_swda_labels/swda_elmo_top_con_out', swda_elmo_top_con_out)
    np.save('model_output_swda_labels/swda_elmo_top_con_out_confs', swda_elmo_top_con_out_confs)
else:
    swda_elmo_con_out = np.load('model_output_swda_labels/swda_elmo_con_out.npy')
    swda_elmo_non_con_out = np.load('model_output_swda_labels/swda_elmo_non_con_out.npy')
    swda_elmo_con_out_confs = np.load('model_output_swda_labels/swda_elmo_con_out_confs.npy')
    swda_elmo_non_con_out_confs = np.load('model_output_swda_labels/swda_elmo_non_con_out_confs.npy')
    swda_elmo_top_con_out = np.load('model_output_swda_labels/swda_elmo_top_con_out.npy')
    swda_elmo_top_con_out_confs = np.load('model_output_swda_labels/swda_elmo_top_con_out_confs.npy')

# Predict with normal elmo mean features
if predict_with_elmo_mean:
    from main_swda_elmo_mean import predict_classes_for_elmo_mean

    swda_elmo_features_mean = np.array([item.mean(axis=0) for item in swda_elmo_features])
    swda_elmo_mean_non_con_out, swda_elmo_mean_con_out, swda_elmo_mean_non_con_out_confs, \
    swda_elmo_mean_con_out_confs = predict_classes_for_elmo_mean(swda_elmo_features_mean)

    np.save('model_output_swda_labels/swda_elmo_mean_con_out', swda_elmo_mean_con_out)
    np.save('model_output_swda_labels/swda_elmo_mean_non_con_out', swda_elmo_mean_non_con_out)
    np.save('model_output_swda_labels/swda_elmo_mean_con_out_confs', swda_elmo_mean_con_out_confs)
    np.save('model_output_swda_labels/swda_elmo_mean_non_con_out_confs', swda_elmo_mean_non_con_out_confs)
else:
    swda_elmo_mean_con_out = np.load('model_output_swda_labels/swda_elmo_mean_con_out.npy')
    swda_elmo_mean_non_con_out = np.load('model_output_swda_labels/swda_elmo_mean_non_con_out.npy')
    swda_elmo_mean_con_out_confs = np.load('model_output_swda_labels/swda_elmo_mean_con_out_confs.npy')
    swda_elmo_mean_non_con_out_confs = np.load('model_output_swda_labels/swda_elmo_mean_non_con_out_confs.npy')

# Evaluation of context model predictions
print('Accuracy comparision between context-based predictions: {}'.format(
    classification.accuracy_score(swda_elmo_con_out, swda_elmo_mean_con_out)))
print('Accuracy with ground truths: {}'.format(
    classification.accuracy_score(swda_elmo_con_out, Ztest)))
print('Kappa (Cohen) score between context-based predictions: {}'.format(
    classification.cohen_kappa_score(swda_elmo_con_out, swda_elmo_mean_con_out)))
print(classification.classification_report(swda_elmo_con_out, swda_elmo_mean_con_out))
print('Spearman Correlation between context-based predictions: {}'.format(
    stats.spearmanr(swda_elmo_con_out, swda_elmo_mean_con_out)))
reliability_data = convert_predictions_to_indices(swda_elmo_con_out, swda_elmo_non_con_out, swda_elmo_mean_con_out,
                                                  swda_elmo_mean_non_con_out, swda_elmo_top_con_out, tags)
k_alpha = alpha(reliability_data, level_of_measurement='nominal')
print("Krippendorff's alpha: {}".format(round(k_alpha, 6)))

fleiss_kappa_score = fleiss_kappa(reliability_data, 5)

print('Accuracy comparision between context and non-context predictions elmo: {}% elmo_mean: {}% '
      'context-context: {}% non-non-context: {}%'.format(
    classification.accuracy_score(swda_elmo_con_out, swda_elmo_non_con_out),
    classification.accuracy_score(swda_elmo_mean_con_out, swda_elmo_mean_non_con_out),
    classification.accuracy_score(swda_elmo_mean_con_out, swda_elmo_con_out),
    classification.accuracy_score(swda_elmo_mean_non_con_out, swda_elmo_non_con_out)))

# Generate final file of annotations; contains "xx" label for unknown/corrections of EDAs
row = ensemble_eda_annotation(swda_elmo_non_con_out, swda_elmo_mean_non_con_out,
                              swda_elmo_con_out, swda_elmo_mean_con_out, swda_elmo_top_con_out,
                              swda_elmo_non_con_out_confs, swda_elmo_mean_non_con_out_confs,
                              swda_elmo_con_out_confs, swda_elmo_mean_con_out_confs, swda_elmo_top_con_out_confs,
                              speaker_id, utterances, speaker_id, emotion,
                              sentiment_labels=[], meld_data=False,
                              file_name='like_swda_final_annotation', write_final_csv=True, write_utterances=True)

final_DA = [item['EDA'] for item in row]
accuracy = classification.accuracy_score(final_DA, Ztest)
print("Accuracy compared to ground truth: {}".format(round(accuracy, 4)))

print('ran swda_dia_act_annotate.py, with total {} number of utterances'.format(len(utterances)))
