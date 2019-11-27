from collections import Counter
from diswiz.utils_server import EDAs
from plot_utils import *
from read_annotated_data_utils import read_data
import numpy as np
import os

utt_Speaker, utt, utt_Emotion, utt_EDAs = read_data('annotated_data/eda_iemocap_dataset.csv')
colors_emo = ['Green', 'Cyan', 'Blue', 'Black', 'Gray', 'Olive', 'Mediumvioletred', 'Orange', 'Red', 'White']
emotions = ['hap', 'exc', 'sur', 'neu', 'xxx', 'fea', 'sad', 'fru', 'ang', 'White']
colors_sent = ['Limegreen', 'Black', 'Darkorange', 'White']
sentiments = ['positive', 'neutral', 'negative', 'White']

tags = sorted(list(Counter(utt_EDAs).keys()))

c = Counter(utt_EDAs)
x = [(i, c[i] / len(utt_EDAs) * 100.0) for i, count in c.most_common()]
for item in x:
    print(item[0], round(item[1], 2))

stack_emotions_values = []
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
    # plot_eda_usage(emotions, pass_values, title, colors_emo, sentiments, sentiments,
    #             colors_sent, data_name='iemocap_bars', plot_pie=False)

plot_bars_plot(stack_emotions_values, emotions[0:9], colors_emo, tags,
               test_show_plot=False, data='iemocap', type_of='emotion')  # , save_eps=True)

print('ran read_annotated_iemocap_data.py')
