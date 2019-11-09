import pandas as pd

train_data = 'MELD/data/MELD/train_sent_emo.csv'
dev_data = 'MELD/data/MELD/dev_sent_emo.csv'
test_data = 'MELD/data/MELD/test_sent_emo.csv'


def read_data_file(data):
    df_train = pd.read_csv(data)  # load the .csv file, specify the appropriate path
    utt = df_train['Utterance'].tolist()  # load the list of utterances
    dia_id = df_train['Dialogue_ID'].tolist()  # load the list of dialogue id's
    utt_id = df_train['Utterance_ID'].tolist()  # load the list of utterance id's
    utt_Emotion = df_train['Emotion'].tolist()
    utt_Sentiment = df_train['Sentiment'].tolist()
    return utt, dia_id, utt_id, utt_Emotion, utt_Sentiment


utt_train_data, dia_id_train_data, utt_id_train_data, \
utt_Emotion_train_data, utt_Sentiment_train_data = read_data_file(train_data)
utt_dev_data, dia_id_dev_data, utt_id_dev_data, \
utt_Emotion_dev_data, utt_Sentiment_dev_data = read_data_file(dev_data)
utt_test_data, dia_id_test_data, utt_id_test_data, \
utt_Emotion_test_data, utt_Sentiment_test_data = read_data_file(test_data)

# for i in range(len(utt)):
#     print ('Utterance: ' + utt[i]) # display utterance
#     print ('Video Path: train_splits/dia' + str(dia_id[i]) + '_utt' + str(utt_id[i]) + '.mp4') # display the video file path
#     print ()
