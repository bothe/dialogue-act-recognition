import pandas as pd

train_data = 'MELD/data/MELD/train_sent_emo.csv'
dev_data = 'MELD/data/MELD/dev_sent_emo.csv'
test_data = 'MELD/data/MELD/test_sent_emo.csv'


def read_data_file(data, annotated_meld=False):
    df_train = pd.read_csv(data)  # load the .csv file, specify the appropriate path
    utt = df_train['Utterance'].tolist()  # load the list of utterances
    dia_id = df_train['Dialogue_ID'].tolist()  # load the list of dialogue id's
    utt_id = df_train['Utterance_ID'].tolist()  # load the list of utterance id's
    utt_emotion = df_train['Emotion'].tolist()
    utt_sentiment = df_train['Sentiment'].tolist()
    utt_speaker = df_train['Speaker'].tolist()
    return utt, dia_id, utt_id, utt_emotion, utt_sentiment, utt_speaker


utt_train_data, dia_id_train_data, utt_id_train_data, \
utt_emotion_train_data, utt_sentiment_train_data, utt_speaker_train = read_data_file(train_data)
utt_dev_data, dia_id_dev_data, utt_id_dev_data, \
utt_emotion_dev_data, utt_sentiment_dev_data, utt_speaker_dev = read_data_file(dev_data)
utt_test_data, dia_id_test_data, utt_id_test_data, \
utt_emotion_test_data, utt_sentiment_test_data, utt_speaker_test = read_data_file(test_data)

# for i in range(len(utt)):
#     print ('Utterance: ' + utt[i]) # display utterance
#     print ('Video Path: train_splits/dia' + str(dia_id[i]) + '_utt' + str(utt_id[i]) + '.mp4') # display the video file path
#     print ()
