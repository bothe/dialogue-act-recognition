import csv

import numpy as np
from scipy.stats import stats
from sklearn.metrics import classification

from MELD.utils.read_meld import *

elmo_feature_retrieval = False
predict_with_elmo = False
predict_with_elmo_mean = False
write_final_csv = False

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
    from main_predictor_online import predict_classes_from_features

    non_con_out, con_out, non_con_out_nums, con_out_nums = predict_classes_from_features(meld_elmo_features_train)
    con_out[0] = non_con_out[0]
    con_out[1] = non_con_out[1]
    np.save('results/meld_con_out_elmo', con_out)
    np.save('results/meld_non_con_out_elmo', non_con_out)
else:
    con_out = np.load('results/meld_con_out_elmo.npy')
    non_con_out = np.load('results/meld_non_con_out_elmo.npy')

if predict_with_elmo_mean:
    meld_elmo_features_test_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_test])
    meld_elmo_features_dev_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_dev])
    meld_elmo_features_train_mean = np.array([item.mean(axis=0) for item in meld_elmo_features_train])

    from main_swda_elmo_mean import *

    meld_elmo_features_train_mean_con = prepare_data(meld_elmo_features_train_mean, [], seq_length, with_y=False)
    mean_con_predictions = context_model.predict(meld_elmo_features_train_mean_con)
    con_out_mean = []
    for item in mean_con_predictions:
        con_out_mean.append(tags[np.argmax(item)])
    con_out_mean.insert(0, non_con_out[1])
    con_out_mean.insert(0, non_con_out[0])
    np.save("results/con_out_mean", con_out_mean)
else:
    con_out_mean = np.load("results/con_out_mean.npy")

print('Accuracy comparision between context- and non-context-based prediction: {}'.format(
    classification.accuracy_score(con_out, con_out_mean)))

print('Kappa (Cohen) score between context- and non-context-based prediction: {}'.format(
    classification.cohen_kappa_score(con_out, con_out_mean)))

print(classification.classification_report(con_out, con_out_mean))

print('Spearman Correlation between context- and non-context-based prediction: {}'.format(
    stats.spearmanr(con_out, con_out_mean)))

if write_final_csv:
    fieldnames = ['speaker', 'uttID', 'utterance', 'emotion',
                  'non_con_out', 'con_out', 'con_out_mean', 'match', 'con_match', 'pre-final-DA', 'all_match']
    store_meld_in_csv = open('results/eda_meld_dataset.csv', mode='w', newline='')
    writer = csv.DictWriter(store_meld_in_csv, fieldnames=fieldnames)
    writer.writeheader()

total_match = 0
con_matches = 0
any_two_matches = 0
utt_info_rows = []
for i in range(len(con_out)):
    match = "NotMatch"
    con_match = "NotConMatch"
    all_match = "NotMatch"
    matched_element = ''
    if con_out_mean[i] == con_out[i] == non_con_out[i]:
        all_match = "AllMatch"
        total_match += 1
    if con_out_mean[i] == con_out[i]:
        con_match = "ConMatch"
        con_matches += 1
    if con_out[i] == non_con_out[i] or con_out_mean[i] == con_out[i] or con_out_mean[i] == non_con_out[i]:
        any_two_matches += 1
        match = "Match"
        if con_out[i] == non_con_out[i]:
            matched_element = con_out[i]
        elif con_out_mean[i] == con_out[i]:
            matched_element = con_out[i]
        elif con_out_mean[i] == non_con_out[i]:
            matched_element = con_out_mean[i]
        utt_info_row = {'speaker': utt_Speaker_train[i].encode("utf-8"),
                        'uttID': utt_id_train_data[i],
                        'utterance': utt_train_data[i].encode("utf-8"),
                        'emotion': utt_Emotion_train_data[i],
                        'non_con_out': str(non_con_out[i]), 'con_out': str(con_out[i]),
                        'con_out_mean': str(con_out_mean[i]), 'pre-final-DA': matched_element,
                        'match': match, 'con_match': con_match, 'all_match': all_match}
    if write_final_csv:
        writer.writerow(utt_info_row)
    utt_info_rows.append(utt_info_row)

print("Matches in all lists(3): {}% and in context lists(2): {}%, any two matches: {}%".format(
    round((total_match / (i + 1)) * 100, 2), round((con_matches / (i + 1)) * 100, 2),
    round((any_two_matches / (i + 1)) * 100, 2)))

print('debug')
