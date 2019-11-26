from collections import Counter
from diswiz.utils_server import EDAs
from mocap_data_reader import get_mocap_data
from plot_utils import *
from read_annotated_data_utils import read_data
import numpy as np

utterances, emotion, emo_evo, v, a, d, speaker_id = get_mocap_data(read_from_csv=True)

utt_Speaker, utt, utt_Emotion, utt_EDAs = read_data('annotated_data/eda_iemocap_dataset.csv')

colors_emo = ['Green', 'Cyan', 'Blue', 'Olive', 'Black', 'Gray', 'Mediumvioletred',  'Orangered', 'Red', 'White']
emotions = ['hap',     'exc',  'sur',  'fea',    'neu',   'xxx',     'sad',            'fru',     'ang', 'White']
# emotions = list(Counter(utt_Emotion).keys())
colors_sent = ['Limegreen', 'Black', 'Darkorange', 'White']
sentiments =['positive', 'neutral', 'negative', 'White']

tags = list(Counter(utt_EDAs).keys())
c = Counter(utt_EDAs)
x = [(i, c[i] / len(utt_EDAs) * 100.0) for i, count in c.most_common()]
for item in x:
    print(item[0], round(item[1], 2))

stack_emotions_values = []
i = 0
for items in emo_evo:
    if len(items) >= 3:
        i += 1

stack_eda = []
pass_emotions, pass_values = [], []
stack_da_names = []
for tag in tags:
    if tag is not str:
        pass
    temp_emotion = []
    for i in range(len(utt)):
        if str(utt_EDAs[i]) == str(tag):
            temp_emotion.append(utt_Emotion[i])
    data_emotion = Counter(temp_emotion)
    values_emotion = []
    # for emotion in emotions[0:9]:
    #     if emotion == 'White':
    #         pass
    #     else:
    #         values_emotion.append(data_emotion[emotion])
    for emotion in emotions[0:9]:
        values_emotion.append(data_emotion[emotion])

    try:
        if tag == 'xx':
            title = tag + '\n' + "Unknown EDA"
        elif tag == 'fo_o_fw_"_by_bc':
            tag = 'fo'
            title = tag + '\n' + EDAs[tag]
        else:
            title = tag + '\n' + EDAs[tag]
    except TypeError:
        title = str(tag)

    pass_emotions = emotions
    pass_values = values_emotion
    pass_values.extend([sum(values_emotion)])

    stack_emotions_values.append(values_emotion)
    stack_da_names.append(tag)

    stack_eda.append(values_emotion)
    # pass_values.extend([sum(values_emotion)])
    # plot_normal_bars(emotions, values, title)
    plot_eda_usage(emotions, pass_values, title, colors_emo, sentiments, sentiments,
                   colors_sent, data_name='iemocap', plot_pie=True)


stack_emo_names = {}
das_stacked = np.array(stack_emotions_values).transpose()
for i in range(len(emotions)):
    stack_emo_names[emotions[i]] = das_stacked[i]
totals = das_stacked.sum(axis=0)


stack_emo_bars = []
for key in stack_emo_names.keys():
    stack_emo_bars.append([i / j * 100 for i, j in zip(stack_emo_names[key], totals)])

r = np.arange(len(stack_da_names))

print('dd')
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
