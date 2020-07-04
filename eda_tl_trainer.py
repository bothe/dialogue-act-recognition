import os
import pickle
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import classification

from models import context_model_att, model_attention_applied_after_bilstm, context_model_att_with_pt_encoder
from src.utils import *
from src.utils import categorize_raw_data

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
seq_length = 3
X_test_con, Y_test_con = prepare_data(X_Test, target_category_test, seq_length)

# NON-CONTEXT MODEL
encoder_model = model_attention_applied_after_bilstm(max_seq_len, X_Test.shape[2], len(tags))
encoder_model.load_weights('params/weight_parameters')
encoder_model_preds = encoder_model.predict(X_Test)[2:].argmax(axis=1)

# CONTEXT MODEL
top_context_model = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(tags), train_with_mean=True)
top_con_model_name = 'params/top_context_model_att_{}'.format(seq_length)
top_context_model.load_weights(top_con_model_name)
top_context_model_swda_preds = top_context_model.predict(X_test_con).argmax(axis=1)

# New CONTEXT MODEL
cm_att_with_pt_encoder = context_model_att_with_pt_encoder(seq_length, max_seq_len, X_test_con.shape[3], len(tags),
                                                           encoder_model)

con_model_name = 'params/context_model_3_pt_encoder'
if os.path.exists(con_model_name):
    cm_att_with_pt_encoder.load_weights(con_model_name)
    old_acc = cm_att_with_pt_encoder.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)[1]
    print('Old Context Model Score results: ', old_acc)
    inter_model_acc = classification.accuracy_score(cm_att_with_pt_encoder.predict(X_test_con).argmax(axis=1),
                                                    top_context_model_swda_preds)
    print('Inter Context Models Score results: ', inter_model_acc)
    inter_non_model_acc = classification.accuracy_score(cm_att_with_pt_encoder.predict(X_test_con).argmax(axis=1),
                                                        encoder_model_preds)
    print('Inter Non and With Context Models Score results: ', inter_non_model_acc)
else:
    print('No model parameters found, model will be trained from start.')
    old_acc = 0.0

train_con_model = True
if train_con_model:
    for i in range(8):
        i += 1
        file_name = 'features/X_train_elmo_features_{}.npy'.format(i)
        print('Loading ', file_name)
        # X_Train.extend(np.load('X_train_elmo_features_{}.npy'.format(i)))
        X_Train = np.load(file_name, allow_pickle=True)  # [0:10000]
        print(X_Train.shape)
        X_Train = padSequencesKeras(np.array(X_Train), max_seq_len, toPadding)
        target = Y_train[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
        target_category_train = to_categorical(target, 42)
        print(X_Train.shape, target_category_train.shape)

        X_Train, Y_train_con = prepare_data(X_Train, target_category_train, seq_length)
        print(X_Train.shape, Y_train_con.shape)

        logdir = "logs/scalars/" + 'cmreps_' + str(time.time()).split('.')[0]
        callbacks_con = [EarlyStopping(patience=3), ModelCheckpoint(filepath=con_model_name, save_best_only=True)]

        cm_att_with_pt_encoder.fit(X_Train, Y_train_con, epochs=5, verbose=2, validation_split=0.20,
                                   callbacks=callbacks_con)

        # cm_att_with_pt_encoder.load_weights(con_model_name)
        evaluation = cm_att_with_pt_encoder.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
        new_acc = evaluation[1]
        print('Context Model Score results: ', new_acc)
        if old_acc < new_acc:
            cm_att_with_pt_encoder.save_weights(con_model_name)
            print('Weights are saved with {} % acc while old acc was {} %'.format(new_acc, old_acc))
            old_acc = new_acc
        evaluation = cm_att_with_pt_encoder.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
        print('Confirm test results for new context model: {}'.format(evaluation[1]))
        X_Train = []

print('debug')
