from keras.utils import to_categorical
import pickle, os, time
from models import model_attention_applied_after_bilstm, non_context_model_for_utterance_level
from training_utils import keras_callbacks
from utils import *


train = False
max_seq_len = 20
con_seq_length = 3
trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'

SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
x_test = pickle.load(open("features/x_test_tokens.p", "rb"))
x_train = pickle.load(open("features/x_train_tokens.p", "rb"))
toPadding = np.load('features/pad_a_token.npy')

tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
target_category_test = to_categorical(Y_test, len(tags))
target_category_train = to_categorical(Y_train, len(tags))

# NON-CONTEXT MODEL
X_test = np.load('features/X_test_elmo_mean_features.npy')
non_context_model = non_context_model_for_utterance_level(X_test.shape[1], len(tags))
non_con_model_name = 'params/swda_elmo_mean_non_con'

# CONTEXT MODEL
X_test_con, Y_test_con = prepare_data(X_test, target_category_test, con_seq_length)
context_model = model_attention_applied_after_bilstm(con_seq_length, X_test_con.shape[2], len(tags))
con_model_name = 'params/context_model_att_elmoMean_{}'.format(con_seq_length)

if train:
    X_train = np.load('features/X_train_elmo_mean_features.npy')
    val_start = int(len(X_train) * 0.2)
    # train non-context model
    logdir = "logs/scalars/" +  'semnc' + str(time.time()).split('.')[0]
    callbacks = keras_callbacks(non_con_model_name, logdir)
    if os.path.exists(non_con_model_name):
        non_context_model.load_weights(non_con_model_name)
    non_context_model.fit(X_train, target_category_train, epochs=50, verbose=2,
                          # X_train[val_start:], target_category_train[val_start:], epochs=50, verbose=2,
                          # validation_data=(X_train[:val_start], target_category_train[:val_start]),  # takes first 20%
                          validation_split=0.15,  # takes last 20%; anyway this worked better
                          callbacks=callbacks)
    non_context_model.load_weights(non_con_model_name)
    loss, old_acc = non_context_model.evaluate(X_test, target_category_test, verbose=2, batch_size=32)
    print('Test results for non-context model:', old_acc)

    # train context model
    logdir = "logs/scalars/" + 'semc' + str(time.time()).split('.')[0]
    callbacks = keras_callbacks(con_model_name, logdir)
    X_train_con, Y_train_con = prepare_data(X_train, target_category_train, con_seq_length)
    context_model.fit(X_train_con, Y_train_con, epochs=50, verbose=2, validation_split=0.20,
                      callbacks=callbacks)

    context_model.load_weights(con_model_name)
    evaluation = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
    print('Context Score results: {}'.format(evaluation[1]))
else:
    non_context_model.load_weights(non_con_model_name)
    evaluation = non_context_model.evaluate(X_test, target_category_test, verbose=2)
    print("Test results for non-context model - accuracy: {}".format(evaluation[1]))

    context_model.load_weights(con_model_name)
    evaluation = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
    print('Context Score results: {}'.format(evaluation[1]))
