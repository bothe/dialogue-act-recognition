from MELD.utils.read_meld import read_data_file
from collections import Counter
import matplotlib.pyplot as plt
from diswiz.utils_server import highDAClass, DAs


utt, utt_id, utt_Emotion, utt_EDAs, utt_Speaker = read_data_file('results/eda_meld_dataset_bk.csv',
                                                                 annotated_meld=True)

emotions = ['joy', 'anger', 'disgust', 'sadness', 'surprise', 'fear', 'neutral']
tags = list(Counter(utt_EDAs).keys())
for tag in tags:
    if tag is not str:
        pass
    temp_emotion = []
    for i in range(len(utt)):
        if utt_EDAs[i] == tag:
            temp_emotion.append(utt_Emotion[i])
    try:
        title = "DA: " + tag.split('"')[0] + " - " + highDAClass(tag, DAs)
    except:
        title = "DA: " + tag

    data = Counter(temp_emotion)
    values = []
    for emotion in emotions:
        values.append(data[emotion])
    plt.bar(emotions, values)
    plt.title(title)
    # plt.ylim(.5, 5.5)
    # plt.xlim(.5, 5.5)
    plt.xlabel('Emotions')
    plt.ylabel('Number of Utterances')
    plt.savefig('figures/meld/fig_' + tag)
    plt.close()
