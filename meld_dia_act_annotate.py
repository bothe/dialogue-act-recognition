import numpy as np
from krippendorff import alpha
from scipy.stats import stats
from sklearn.metrics import classification
import os
from MELD.utils.read_meld import *
from final_annotator_utils import ensemble_annotation, convert_predictions_to_indices

elmo_feature_retrieval = False
predict_with_elmo = True
predict_with_elmo_mean = True

if os.path.exists('results/tags.npy'):
    tags = np.load('results/tags.npy')

if elmo_feature_retrieval:
    from elmo_features import get_elmo_fea

    meld_elmo_features_test = get_elmo_fea(utt_test_data, mean=False)
    np.save('features/meld_elmo_features_test', meld_elmo_features_test)
    meld_elmo_features_dev = get_elmo_fea(utt_dev_data, mean=False)
    np.save('features/meld_elmo_features_dev', meld_elmo_features_dev)
    meld_elmo_features_train = get_elmo_fea(utt_train_data, mean=False)
    np.save('features/meld_elmo_features_train', meld_elmo_features_train)
elif predict_with_elmo or predict_with_elmo_mean:
    meld_elmo_features_test = np.load('features/meld_elmo_features_test.npy', allow_pickle=True)
    meld_elmo_features_dev = np.load('features/meld_elmo_features_dev.npy', allow_pickle=True)
    meld_elmo_features_train = np.load('features/meld_elmo_features_train.npy', allow_pickle=True)

if predict_with_elmo:
    from main_predictor_online import predict_classes_for_elmo

    meld_non_con_out_elmo, meld_con_out_elmo, non_con_out_confs, con_out_confs = predict_classes_for_elmo(
        meld_elmo_features_train)
    np.save('results/meld_con_out_elmo', meld_con_out_elmo)
    np.save('results/meld_non_con_out_elmo', meld_non_con_out_elmo)
    np.save('results/meld_con_out_elmo_confs', con_out_confs)
    np.save('results/meld_non_con_out_elmo_confs', non_con_out_confs)
else:
    meld_con_out_elmo = np.load('results/meld_con_out_elmo.npy')
    meld_non_con_out_elmo = np.load('results/meld_non_con_out_elmo.npy')
    con_out_confs = np.load('results/meld_con_out_elmo_confs')
    non_con_out_confs = np.load('results/meld_non_con_out_elmo_confs')

if predict_with_elmo_mean:
    meld_elmo_features_test_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_test])
    meld_elmo_features_dev_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_dev])
    meld_elmo_features_train_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_train])

    from main_swda_elmo_mean import *

    meld_elmo_features_train_mean_con = prepare_data(meld_elmo_features_train_mean, [], con_seq_length, with_y=False)
    mean_con_predictions = context_model.predict(meld_elmo_features_train_mean_con)
    con_out_mean = []
    for item in mean_con_predictions:
        con_out_mean.append(tags[np.argmax(item)])
    con_out_mean.insert(0, meld_non_con_out_elmo[1])
    con_out_mean.insert(0, meld_non_con_out_elmo[0])
    np.save("results/con_out_mean", con_out_mean)
else:
    con_out_mean = np.load("results/con_out_mean.npy")

# Evaluation of context model predictions
print('Accuracy comparision between context- and non-context-based prediction: {}'.format(
    classification.accuracy_score(meld_con_out_elmo, con_out_mean)))
print('Kappa (Cohen) score between context- and non-context-based prediction: {}'.format(
    classification.cohen_kappa_score(meld_con_out_elmo, con_out_mean)))
print(classification.classification_report(meld_con_out_elmo, con_out_mean))
print('Spearman Correlation between context- and non-context-based prediction: {}'.format(
    stats.spearmanr(meld_con_out_elmo, con_out_mean)))
reliability_data = convert_predictions_to_indices(meld_con_out_elmo, meld_non_con_out_elmo, con_out_mean,
                                                  meld_non_con_out_elmo, tags)
k_alpha = alpha(reliability_data, level_of_measurement='nominal')
print("Krippendorff's alpha: {}".format(round(k_alpha, 6)))

# Generate final file of annotations; contains CORRECT label for corrections in EDAs
row = ensemble_annotation(meld_non_con_out_elmo, meld_con_out_elmo, con_out_mean, utt_Speaker_train, utt_train_data,
                          utt_id_train_data, utt_Emotion_train_data, sentiment_labels=utt_Sentiment_train_data,
                          meld_data=True, file_name='meld_emotion', write_final_csv=False)

print('ran meld_dia_act_annotate.py')
