import pickle

from keras.utils import to_categorical
from sklearn.metrics import classification

from models import context_model_att, model_attention_applied_after_bilstm, context_model_att_with_pt_encoder
from src.read_annotated_data_utils import read_data
from src.utils import *
from src.utils import categorize_raw_data

utt_Speaker, utt, utt_Emotion, utt_EDAs = read_data('annotated_data/eda_iemocap_dataset.csv')
iemocap_elmo_features = np.load('features/iemocap_elmo_features.npy', allow_pickle=True)

test_split = 7869
utt_ids_train, utt_ids_test = utt_Speaker[:test_split], utt_Speaker[test_split:]
utt_train, utt_test = utt[:test_split], utt[test_split:]
utt_Emotion_train, utt_Emotion_test = utt_Emotion[:test_split], utt[test_split:]
utt_EDAs_train, utt_EDAs_test = utt_EDAs[:test_split], utt_EDAs[test_split:]
iemocap_elmo_features_train = iemocap_elmo_features[:test_split]
iemocap_elmo_features_test = iemocap_elmo_features[test_split:]

seq_length = 3
max_seq_len = 20
trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
x_test = pickle.load(open("features/x_test_tokens.p", "rb"))
x_train = pickle.load(open("features/x_train_tokens.p", "rb"))
toPadding = np.load('features/pad_a_token.npy')

X_Test = np.load('features/X_test_elmo_features.npy', allow_pickle=True)
X_Test = padSequencesKeras(X_Test, max_seq_len, toPadding)
tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
target_category_test = to_categorical(Y_test, len(tags))

# Preparing test data for contextual training with Seq_length
X_test_con, Y_test_con = prepare_data(X_Test, target_category_test, seq_length)

iemocap_elmo_features_train = padSequencesKeras(iemocap_elmo_features_train, max_seq_len, toPadding)
iemocap_elmo_features_train_con = prepare_data(iemocap_elmo_features_train, [], seq_length, with_y=False)

iemocap_elmo_features_test = padSequencesKeras(iemocap_elmo_features_test, max_seq_len, toPadding)
iemocap_elmo_features_test_con = prepare_data(iemocap_elmo_features_test, [], seq_length, with_y=False)

# NON-CONTEXT MODEL
model = model_attention_applied_after_bilstm(max_seq_len, X_Test.shape[2], len(tags))
model.load_weights('params/weight_parameters')

# CONTEXT MODEL
top_context_model = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(tags), train_with_mean=True)
top_con_model_name = 'params/top_context_model_att_{}'.format(seq_length)
top_context_model.load_weights(top_con_model_name)

# CONTEXT MODEL with Pre-trained encoder
cm_att_with_pt_encoder = context_model_att_with_pt_encoder(seq_length, max_seq_len, X_test_con.shape[3],
                                                           len(tags), model)
con_model_name = 'params/context_model_3_pt_encoder'
cm_att_with_pt_encoder.load_weights(con_model_name)


non_con_predictions = model.predict(iemocap_elmo_features_test)
top_con_predictions = top_context_model.predict(iemocap_elmo_features_test_con)
cm_att_with_pt_encoder_preds = cm_att_with_pt_encoder.predict(iemocap_elmo_features_test_con)

iemocap_acc = classification.accuracy_score(cm_att_with_pt_encoder_preds.argmax(axis=1),
                                            top_con_predictions.argmax(axis=1))
print('Inter Context Models Accuracy ', iemocap_acc)
print('debug')
