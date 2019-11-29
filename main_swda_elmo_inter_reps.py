import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Model
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

# CONTEXT MODEL ELMO MEAN
context_model_elmo_mean = model_attention_applied_after_bilstm(con_seq_length, X_test_mean.shape[2], len(tags))
con_model_name_mean = 'params/context_model_att_elmoMean_{}'.format(con_seq_length)
context_model_elmo_mean.load_weights(con_model_name_mean)
loss, old_acc = context_model_elmo_mean.evaluate(X_test_mean, Y_test_con, verbose=2, batch_size=32)
print('Context Score results:', old_acc)

if get_inter_reps_from_model:
    max_seq_len = 20
    X_Test = np.load('features/X_test_elmo_features.npy', allow_pickle=True)
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
        X_Train = np.load('features/X_train_elmo_features_{}.npy'.format(i), allow_pickle=True)
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
    pass
    # intermediate_output_x_test = np.load('features/swda_inter_reps/X_test_elmo_features_flatten_attention.npy')
    # intermediate_output_x_train = np.load('features/swda_inter_reps/X_train_elmo_features_flatten_attention.npy')

if train_con_model:
    X_train_mean = np.load('features/X_train_elmo_mean_features.npy')
    # old_acc = 0.0
    for i in range(8):
        i += 1
        file_name = 'features/X_train_elmo_features_{}.npy'.format(i)
        print('Reding ', file_name)
        # X_Train.extend(np.load('X_train_elmo_features_{}.npy'.format(i)))
        X_Train = np.load(file_name, allow_pickle=True)[0:5000]
        print(X_Train.shape)
        X_Train = padSequencesKeras(np.array(X_Train), max_seq_len, toPadding)
        target = Y_train[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
        target_category_train = to_categorical(target, 42)
        print(X_Train.shape, target_category_train.shape)

        X_Train, Y_train_con = prepare_data(X_Train, target_category_train, con_seq_length)
        print(X_Train.shape, Y_train_con.shape)


        # train non-context model
        con_model_name = 'params/context_model_3_reps'
        logdir = "logs/scalars/" + 'cmreps_' + str(time.time()).split('.')[0]
        callbacks_con = [EarlyStopping(patience=3), ModelCheckpoint(filepath=con_model_name, save_best_only=True)]
        context_model_reps = model_attention_applied_after_bilstm(con_seq_length, X_test_con.shape[2], len(tags))
        if os.path.exists(con_model_name):
            context_model_reps.load_weights(con_model_name)
        else:
            print('No model parameters found, model will be trained from start.')

        context_model_reps.fit(X_Train, Y_train_con, epochs=50, verbose=2, validation_split=0.20,
                               callbacks=callbacks_con)

        context_model_reps.load_weights(con_model_name)
        evaluation = context_model_reps.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
        new_acc = evaluation[1]
        print('Context Score results: ', new_acc)
        if old_acc < new_acc:
            context_model_reps.save_weights(con_model_name)
            print('Weights are saved with {} % acc while old acc was {} %'.format(new_acc, old_acc))
            old_acc = new_acc
        evaluation = context_model_reps.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
        print('Test results for context model: {}'.format(evaluation[1]))
        X_Train = 0
