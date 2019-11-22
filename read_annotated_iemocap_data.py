from collections import Counter
from diswiz.utils_server import EDAs
from mocap_data_reader import get_mocap_data
from plot_utils import *
from read_annotated_data_utils import read_data

# utterances, emotion, emo_evo, v, a, d, speaker_id = get_mocap_data()

utt_Speaker, utt, utt_Emotion, utt_EDAs = read_data('results/eda_iemocap_dataset.csv')

colors_emo = ['Green', 'Blue', 'Olive', 'Mediumvioletred', 'Black', 'Gray', 'Orangered',  'Yellow', 'Red', 'White']
emotions = ['joy', 'sadness', 'fear', 'neutral', 'surprise', 'disgust', 'anger', 'White']
emotions = ['hap',      'exc',   'sad',       'fea',         'neu', 'xxx',    'sur',      'fru',   'ang', 'White']
# emotions = list(Counter(utt_Emotion).keys())
colors_sent = ['Limegreen', 'Black', 'Darkorange', 'White']
sentiments =['positive', 'neutral', 'negative', 'White']

tags = list(Counter(utt_EDAs).keys())
c = Counter(utt_EDAs)
x = [(i, c[i] / len(utt_EDAs) * 100.0) for i, count in c.most_common()]
for item in x:
    print(item[0], round(item[1], 2))

stack_emotions_values = []
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
        title = tag + '\n' + EDAs[tag]
    except TypeError:
        title = str(tag)

    pass_emotions = emotions
    pass_values = values_emotion
    pass_values.extend([sum(values_emotion)])

    stack_emotions_values.append(values_emotion)
    stack_da_names.append(tag)

    # plot_normal_bars(emotions, values, title)
    # plot_eda_usage(emotions, pass_values, title, colors_emo, sentiments, sentiments,
                 #   colors_sent, data_name='iemocap')

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
