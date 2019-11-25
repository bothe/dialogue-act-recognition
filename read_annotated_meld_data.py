from MELD.utils.read_meld import read_data_file
from collections import Counter
from diswiz.utils_server import EDAs
from plot_utils import *

utt, utt_id, utt_Emotion, utt_EDAs, utt_Speaker = read_data_file('results/eda_meld_dataset_bk.csv',
                                                                 annotated_meld=True)
train_data = 'MELD/data/MELD/train_sent_emo.csv'
utt_train_data, dia_id_train_data, utt_id_train_data, \
utt_Emotion_train_data, utt_Sentiment_train_data, utt_Speaker_train = read_data_file(train_data)
colors_emo = ['Green', 'Blue',       'Olive',    'Black',   'Mediumvioletred', 'Orangered', 'Red', 'White']
emotions = ['joy',     'sadness',     'fear',    'neutral',    'surprise',      'disgust', 'anger', 'White']
colors_sent = ['Limegreen', 'Black', 'Darkorange', 'White']
sentiments =['positive', 'neutral', 'negative', 'White']

tags = list(Counter(utt_EDAs).keys())
c = Counter(utt_EDAs)
x = [(i, c[i] / len(utt_EDAs) * 100.0) for i, count in c.most_common()]
for item in x:
    print(item[0], round(item[1], 2))

stack_da_emotions, stack_da_sentiments = {}, {}
pass_emotions, pass_values = [], []
for tag in tags:
    if tag is not str:
        pass
    temp_emotion, temp_sentiment = [], []
    for i in range(len(utt)):
        if str(utt_EDAs[i]) == str(tag):
            temp_emotion.append(utt_Emotion[i])
            temp_sentiment.append(utt_Sentiment_train_data[i])
    data_emotion = Counter(temp_emotion)
    data_sentiment = Counter(temp_sentiment)
    values_emotion, values_sentiment = [], []
    for emotion in emotions:
        values_emotion.append(data_emotion[emotion])
    for sentiment in sentiments:
        values_sentiment.append(data_sentiment[sentiment])

    try:
        title = tag + '\n' + EDAs[tag]
    except TypeError:
        title = str(tag)

    pass_emotions = emotions
    pass_values = values_emotion
    pass_values.extend([sum(values_emotion)])

    pass_values_sent = values_sentiment
    pass_values_sent.extend([sum(values_sentiment)])
    # plot_normal_bars(emotions, values, title)
    # plot_eda_usage(emotions, pass_values, title, colors_emo, sentiments, pass_values_sent, colors_sent)
    stack_da_emotions[title] = values_emotion
    stack_da_sentiments[title] = values_sentiment
