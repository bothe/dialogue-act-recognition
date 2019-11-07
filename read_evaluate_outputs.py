from sklearn.metrics import classification
from collections import Counter
from mocap_annotator import get_mocap_data
from results.read_predictions_utils import labels_to_indices, read_all_predictions
import matplotlib.pyplot as plt


utterances, emotion, emo_evo, v, a, d = get_mocap_data()
con_elmo_embs, con_diswiz, non_con_elmo_embs, non_con_diswiz = read_all_predictions()
all_classes = Counter(list(con_diswiz) + list(con_elmo_embs) + list(non_con_diswiz) + list(non_con_elmo_embs))
classes = list(all_classes.keys())

dict_of_all_info = {}
for item in classes:
    v_temp, a_temp = [], []
    for i in range(len(con_elmo_embs)):
        if con_elmo_embs[i] == "fc":
            v_temp.append(v[i])
            a_temp.append(a[i])
    dict_of_all_info[item] = {"v": v_temp, "a": a_temp, "v_count": Counter(v_temp), "a_count": Counter(a_temp)}
    for key, value in Counter(a_temp):
        pass  # TODO plot the different cases


con_elmo_embs = labels_to_indices(con_elmo_embs, classes)
con_diswiz = labels_to_indices(con_diswiz, classes)
non_con_elmo_embs = labels_to_indices(non_con_elmo_embs, classes)
non_con_diswiz = labels_to_indices(non_con_diswiz, classes)

print("accuracy_score: \n", classification.accuracy_score(con_elmo_embs, non_con_elmo_embs))
print("matthews_corrcoef: \n", classification.matthews_corrcoef(con_elmo_embs, non_con_elmo_embs))


plt.plot()

print('debug')
