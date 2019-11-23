from collections import Counter
from diswiz.utils_server import EDAs
from mocap_data_reader import get_mocap_data
from plot_utils import *
from read_annotated_data_utils import read_data
import numpy as np

utterances, emotion, emo_evo, v, a, d, speaker_id = get_mocap_data()

utt_Speaker, utt, utt_Emotion, utt_EDAs, utt_EDAs_corrected = read_data('results/eda_iemocap_dataset.csv')

colors_emo = ['Green', 'Cyan', 'Blue', 'Olive', 'Black', 'Gray', 'Mediumvioletred',  'Orangered', 'Red', 'White']
emotions = ['hap',      'exc',   'sad',  'fea',  'neu',   'xxx',    'sur',           'fru',       'ang', 'White']
# emotions = list(Counter(utt_Emotion).keys())
colors_sent = ['Limegreen', 'Black', 'Darkorange', 'White']
sentiments =['positive', 'neutral', 'negative', 'White']

tags = list(Counter(utt_EDAs).keys())
c = Counter(utt_EDAs_corrected)
x = [(i, c[i] / len(utt_EDAs_corrected) * 100.0) for i, count in c.most_common()]
for item in x:
    print(item[0], round(item[1], 2))

i = 0
for items in emo_evo:
    if len(items) >= 3:
        i += 1

stack_eda = []
pass_emotions, pass_values = [], []
for tag in tags:
    if tag is not str:
        pass
    temp_emotion = []
    for i in range(len(utt)):
        if str(utt_EDAs_corrected[i]) == str(tag):
            temp_emotion.append(utt_Emotion[i])
    data_emotion = Counter(temp_emotion)
    values_emotion = []
    for emotion in emotions:
        if emotion == 'White':
            pass
        else:
            values_emotion.append(data_emotion[emotion])

    try:
        title = tag + '\n' + EDAs[tag]
    except TypeError:
        title = str(tag)

    pass_emotions = emotions
    pass_values = values_emotion
    stack_eda.append(values_emotion)
    # pass_values.extend([sum(values_emotion)])
    # plot_normal_bars(emotions, values, title)
    # plot_pie_half_usage(emotions, pass_values, title, colors_emo, sentiments, sentiments,
    #                    colors_sent, data_name='iemocap')

edas_stack = np.array(stack_eda).transpose()
edas_stack_sum = edas_stack.sum(axis=0)
for i in range(edas_stack_sum.shape[0]):
    if edas_stack_sum[i] == 0:
        edas_stack_sum[i] = 1

import pandas as pd
## convert your array into a dataframe
df = pd.DataFrame(edas_stack/edas_stack_sum)
df.to_excel('file.xls', index=False)


print('ran read_annotated_iemocap_data.py')
