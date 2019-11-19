from sklearn.metrics import classification
from collections import Counter
from mocap_annotator import get_mocap_data
from results.read_predictions_utils import labels_to_indices, read_all_predictions
import matplotlib.pyplot as plt
import numpy as np
from diswiz.utils_server import highDAClass, DAs
import csv


utterances, emotion, emo_evo, v, a, d = get_mocap_data()
con_elmo_embs, con_diswiz, non_con_elmo_embs, non_con_diswiz = read_all_predictions()
all_classes = Counter(list(con_elmo_embs) + list(non_con_elmo_embs))
# all_classes = Counter(list(con_diswiz) + list(con_elmo_embs) + list(non_con_diswiz) + list(non_con_elmo_embs))
classes = list(all_classes.keys())
#  classes = list(Counter(con_elmo_embs).keys())
dict_of_all_info = {}
for item in classes:
    v_temp, a_temp = [], []
    v_a_temp = []
    for i in range(len(con_elmo_embs)):
        if con_elmo_embs[i] == item:
            v_temp.append(v[i])
            a_temp.append(a[i])
            v_a_temp.append((v[i], a[i]))
    v_a_temp_count = Counter(v_a_temp)
    dict_of_all_info[item] = {"v": v_temp, "a": a_temp, "v_a_count": v_a_temp_count}
    x, y, z = [], [], []
    for items in list(v_a_temp_count.keys()):
        x.append(items[0])
        y.append(items[1])
        z.append(v_a_temp_count[items])

    try:
        title = "DA: " + item.split('"')[0] + " - " + highDAClass(item, DAs)
    except:
        title = "DA: " + item

    plt.scatter(x, y, np.array(z) * 10, alpha=0.5)
    plt.title(title)
    plt.ylim(.5, 5.5)
    plt.xlim(.5, 5.5)
    plt.xlabel('valence')
    plt.ylabel('arousal')
    plt.savefig('figures/fig_'+item)
    plt.close()

con_elmo_embs = labels_to_indices(con_elmo_embs, classes)
con_diswiz = labels_to_indices(con_diswiz, classes)
non_con_elmo_embs = labels_to_indices(non_con_elmo_embs, classes)
non_con_diswiz = labels_to_indices(non_con_diswiz, classes)

print("accuracy_score: \n", classification.accuracy_score(con_elmo_embs, non_con_elmo_embs))
print("matthews_corrcoef: \n", classification.matthews_corrcoef(con_elmo_embs, non_con_elmo_embs))

print('debug')


def ensemble_annotation(non_con_out, con_out, con_out_mean, meld_data=True):
    write_final_csv = True
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
            if meld_data:
                utt_info_row = {'speaker': utt_Speaker_train[i].encode("utf-8"),
                                'uttID': utt_id_train_data[i],
                                'utterance': utt_train_data[i].encode("utf-8"),
                                'emotion': utt_Emotion_train_data[i],
                                'non_con_out': str(non_con_out[i]), 'con_out': str(con_out[i]),
                                'con_out_mean': str(con_out_mean[i]), 'pre-final-DA': matched_element,
                                'match': match, 'con_match': con_match, 'all_match': all_match}
            else:
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

