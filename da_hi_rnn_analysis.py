import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Model
import tensorflow as tf
from keras.utils import to_categorical
import os
from models import model_attention_applied_after_bilstm, context_model_att
from src.utils import *

get_inter_reps_from_model = False
train_con_model = True
con_seq_length = 3
max_seq_len = 20
trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
toPadding = np.load('features/pad_a_token.npy', allow_pickle=True)
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
target_category_test = to_categorical(Y_test, len(tags))
target_category_train = to_categorical(Y_train, len(tags))

X_Test = np.load('features/X_test_elmo_features.npy', allow_pickle=True)
X_test_mean = np.load('features/X_test_elmo_mean_features.npy', allow_pickle=True)

X_Test = padSequencesKeras(X_Test, max_seq_len, toPadding)
# Preparing for contextual training
X_test_con, Y_test_con = prepare_data(X_Test, target_category_test, con_seq_length)
X_test_mean = prepare_data(X_test_mean, [], con_seq_length, with_y=False)

EMBEDDING_DIM = X_test_con.shape[3]
INPUT_DIM = X_test_con.shape[2]
TIME_STEPS = X_test_con.shape[1]
NUM_CLASSES = Y_test_con.shape[1]


# CONTEXT MODEL ELMO
context_model = context_model_att(con_seq_length, INPUT_DIM, EMBEDDING_DIM, NUM_CLASSES)
con_model_name = 'params/context_model_att_{}'.format(con_seq_length)
context_model.load_weights(con_model_name)
loss, old_acc = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
print('Context Score results:', old_acc)

# NON-CONTEXT MODEL
non_con_model = model_attention_applied_after_bilstm(max_seq_len, X_Test.shape[2], len(tags))
non_con_model.load_weights('params/weight_parameters')
evaluation = non_con_model.evaluate(X_Test, target_category_test, verbose=2)
print("Test results for non-context model - accuracy: {}".format(evaluation[1]))

layer_name = 'flatten_attention'
intermediate_layer_model = Model(inputs=non_con_model.input,
                                 outputs=non_con_model.get_layer(layer_name).output)
intermediate_output_x_test = intermediate_layer_model.predict(X_Test)

print('debug')
