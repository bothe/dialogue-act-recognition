from sklearn.metrics import classification
from collections import Counter
from mocap_data_reader import get_mocap_data
from results.read_predictions_utils import labels_to_indices, read_all_predictions
import matplotlib.pyplot as plt
import numpy as np
from diswiz.utils_server import highDAClass, DAs

utterances, emotion, emo_evo, v, a, d, speaker_id = get_mocap_data()
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
