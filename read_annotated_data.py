from MELD.utils.read_meld import read_data_file
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from diswiz.utils_server import EDAs
from plot_utils import *

utt, utt_id, utt_Emotion, utt_EDAs, utt_Speaker = read_data_file('results/eda_meld_dataset_bk.csv',
                                                                 annotated_meld=True)
colors = ['Green', 'Red', 'Orange', 'Gray', 'Magenta', 'Black', 'Blue', 'White']
emotions = ['joy', 'anger', 'disgust', 'sadness', 'surprise', 'fear', 'neutral']
tags = list(Counter(utt_EDAs).keys())
pass_emotions, pass_values = [], []
for tag in tags:
    if tag is not str:
        pass
    temp_emotion = []
    for i in range(len(utt)):
        if str(utt_EDAs[i]) == str(tag):
            temp_emotion.append(utt_Emotion[i])
    data = Counter(temp_emotion)
    values = []
    for emotion in emotions:
        values.append(data[emotion])

    try:
        title = tag + '\n' + EDAs[tag]
    except TypeError:
        title = str(tag)

    pass_emotions = emotions
    pass_emotions.extend(['White'])
    pass_values = values
    pass_values.extend([sum(values)])
    # plot_normal_bars(emotions, values, title)
    plot_pie_half_usage(pass_emotions, pass_values, title, colors)
