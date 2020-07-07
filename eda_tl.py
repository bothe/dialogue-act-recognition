import pickle

import keras
from keras.utils import to_categorical
from sklearn.metrics import classification

from models import context_model_att, model_attention_applied_after_bilstm, context_model_att_with_pt_encoder
from src.eda_tl_utils import merge_emotion_classes
from src.read_annotated_data_utils import read_data
from src.utils import *
from src.utils import categorize_raw_data

utt_Speaker, utt, utt_Emotion, utt_EDAs = read_data('annotated_data/eda_iemocap_dataset.csv')
iemocap_elmo_features = np.load('features/iemocap_elmo_features.npy', allow_pickle=True)

# Merge the emotion classes
emotion_classes, utt_Emotion_merged, utt_Emotion_merger_ids = merge_emotion_classes(utt_Emotion)

test_split = 7869
utt_ids_train, utt_ids_test = utt_Speaker[:test_split], utt_Speaker[test_split:]
utt_train, utt_test = utt[:test_split], utt[test_split:]
utt_Emotion_train = to_categorical(utt_Emotion_merger_ids[:test_split], len(emotion_classes))
utt_Emotion_test = to_categorical(utt_Emotion_merger_ids[test_split:], len(emotion_classes))
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

# SCONTEXT MODEL
top_context_model = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(tags), train_with_mean=True)
top_con_model_name = 'params/top_context_model_att_{}'.format(seq_length)
top_context_model.load_weights(top_con_model_name)


# EMO-CONTEXT MODEL
emo_context_model = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(emotion_classes), train_with_mean=True)
emo_context_model_name = 'params/emo_context_model_att_{}'.format(seq_length)


# emo_context_model.fit(iemocap_elmo_features_train_con, utt_Emotion_train[2:], epochs=10, verbose=2)
print(emo_context_model.evaluate(iemocap_elmo_features_test_con, utt_Emotion_test[2:]))

inter_output_model = keras.engine.Model(top_context_model.input,
                                        top_context_model.get_layer(index=3).output)
inter_output_model.trainable = False

emo_tl_model = keras.Sequential([
    inter_output_model,
    keras.layers.Dense(len(emotion_classes)),
])
emo_tl_model.summary()
emo_tl_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

emo_tl_model.fit(iemocap_elmo_features_train_con, utt_Emotion_train[2:], epochs=10, verbose=2)
print(emo_tl_model.evaluate(iemocap_elmo_features_test_con, utt_Emotion_test[2:]))

print('debug')
