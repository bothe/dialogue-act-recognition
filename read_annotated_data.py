from MELD.utils.read_meld import read_data_file
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from diswiz.utils_server import EDAs

utt, utt_id, utt_Emotion, utt_EDAs, utt_Speaker = read_data_file('results/eda_meld_dataset_bk.csv',
                                                                 annotated_meld=True)

emotions = ['joy', 'anger', 'disgust', 'sadness', 'surprise', 'fear', 'neutral']
tags = list(Counter(utt_EDAs).keys())
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
        title = tag + ' - ' + EDAs[tag]
    except TypeError:
        title = str(tag)

    plt.rcParams.update({'font.size': 16})
    plt.bar(emotions, values)
    plt.title(title)
    plt.xticks(rotation=15)

    # plt.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.ylim(.5, 5.5)
    # plt.xlim(.5, 5.5)
    # plt.xlabel('Emotions')
    # plt.ylabel('Number of Utterances')
    plt.savefig('figures/meld/fig_' + title)
    plt.close()
