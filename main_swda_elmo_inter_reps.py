import time

from keras import Model
from keras.utils import to_categorical
import pickle, os
from models import model_attention_applied_after_bilstm
from training_utils import keras_callbacks
from utils import *

get_inter_reps = False
train_con_model = True
con_seq_length = 3
trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
target_category_test = to_categorical(Y_test, len(tags))
target_category_train = to_categorical(Y_train, len(tags))


if get_inter_reps:
    max_seq_len = 20
    toPadding = np.load('features/pad_a_token.npy')
    X_Test = np.load('features/X_test_elmo_features.npy')
    X_Test = padSequencesKeras(X_Test, max_seq_len, toPadding)

    # NON-CONTEXT MODEL
    non_con_model = model_attention_applied_after_bilstm(max_seq_len, X_Test.shape[2], len(tags))
    non_con_model.load_weights('params/weight_parameters')
    evaluation = non_con_model.evaluate(X_Test, target_category_test, verbose=2)
    print("Test results for non-context model - accuracy: {}".format(evaluation[1]))

    layer_name = 'flatten_attention'
    intermediate_layer_model = Model(inputs=non_con_model.input,
                                     outputs=non_con_model.get_layer(layer_name).output)

    intermediate_output_x_test = intermediate_layer_model.predict(X_Test)

    intermediate_output_x_train = []
    for i in range(8):
        i += 1
        print('X_train_elmo_features_{}.npy'.format(i))
        # X_Train.extend(np.load('X_train_elmo_features_{}.npy'.format(i)))
        X_Train = np.load('features/X_train_elmo_features_{}.npy'.format(i))
        print(X_Train.shape)
        X_Train = padSequencesKeras(np.array(X_Train), max_seq_len, toPadding)
        target = Y_train[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
        target_category_train = to_categorical(target, 42)
        print(X_Train.shape, target_category_train.shape)

        intermediate_output_x_train_temp = intermediate_layer_model.predict(X_Train)
        intermediate_output_x_train.extend(intermediate_output_x_train_temp)
        X_Train = 0
    np.save('features/swda_inter_reps/X_test_elmo_features_flatten_attention', intermediate_output_x_test)
    np.save('features/swda_inter_reps/X_train_elmo_features_flatten_attention', intermediate_output_x_train)
else:
    intermediate_output_x_test = np.load('features/swda_inter_reps/X_test_elmo_features_flatten_attention.npy')
    intermediate_output_x_train = np.load('features/swda_inter_reps/X_train_elmo_features_flatten_attention.npy')

if train_con_model:
    X_train = np.load('features/X_train_elmo_mean_features.npy')
    X_test = np.load('features/X_test_elmo_mean_features.npy')

    X_test = np.hstack((X_test, intermediate_output_x_test))
    X_train = np.hstack((X_train, intermediate_output_x_train))

    X_test_con, Y_test_con = prepare_data(X_test, target_category_test, con_seq_length)
    X_train_con, Y_train_con = prepare_data(X_train, target_category_train, con_seq_length)

    # train non-context model
    con_model_name = 'params/context_model_3_reps'
    logdir = "logs/scalars/" + 'cmreps_' + str(time.time()).split('.')[0]
    callbacks = keras_callbacks(con_model_name, logdir)
    context_model_reps = model_attention_applied_after_bilstm(con_seq_length, X_test_con.shape[2], len(tags))
    if os.path.exists(con_model_name):
        context_model_reps.load_weights(con_model_name)
    else:
        print('No model parameters found, model will be trained from start.')

    context_model_reps.fit(X_train_con, Y_train_con, epochs=50, verbose=2, validation_split=0.20,
                           callbacks=callbacks)

    context_model_reps.load_weights(con_model_name)
    evaluation = context_model_reps.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
    print('Test results for context model: {}'.format(evaluation[1]))
