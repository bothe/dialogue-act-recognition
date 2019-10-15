from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import pickle, os
from models import model_attention_applied_after_bilstm, context_model_att
from utils import *

max_seq_len = 20

trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
x_test = pickle.load(open("features/x_test_tokens.p", "rb"))
x_train = pickle.load(open("features/x_train_tokens.p", "rb"))
toPadding = np.load('features/pad_a_token.npy')
X_Test = np.load('features/X_test_elmo_features.npy')
X_Test = padSequencesKeras(X_Test, max_seq_len, toPadding)

tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
target_category_test = to_categorical(Y_test, len(tags))

# NON-CONTEXT MODEL
SINGLE_ATTENTION_VECTOR = False
model = model_attention_applied_after_bilstm(max_seq_len,  X_Test.shape[2], len(tags), SINGLE_ATTENTION_VECTOR)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [ModelCheckpoint(filepath='weight_parameters', save_best_only=True)]  # EarlyStopping(patience=5),
model.load_weights('params/weight_parameters')
evaluation = model.evaluate(X_Test, target_category_test, verbose=2)
print("Test results for non-context model - accuracy: {}".format(evaluation[1]))

# Preparing for contextual training
seq_length = 3
X_test_con, Y_test_con = preparedata(X_Test, target_category_test, seq_length)

EMBEDDING_DIM = X_test_con.shape[3]
INPUT_DIM = X_test_con.shape[2]
TIME_STEPS = X_test_con.shape[1]
NUM_CLASSES = Y_test_con.shape[1]
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False

# CONTEXT MODEL
context_model = context_model_att(seq_length, INPUT_DIM, EMBEDDING_DIM, NUM_CLASSES)
con_model_name = 'params/context_model_att_{}'.format(seq_length)
callbacks_con = [EarlyStopping(patience=3), ModelCheckpoint(filepath=con_model_name, save_best_only=True)]
if os.path.exists(con_model_name):
    context_model.load_weights(con_model_name)
    loss, old_acc = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
    print('Context Score results:', old_acc)

train = True
if train == False:
    loss1, new_acc1  = model.evaluate(X_Test, target_category_test, verbose=2)
    print('Non-Context Score results:', new_acc1)
    loss2, new_acc2 = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
    print('Context Score results:', new_acc2)

else:
    # old_acc = 0
    for iteration in range(10):
        print('Iteration number {}'.format(str(iteration + 1)))
        for i in range(8):
            i += 1
            print('X_train_elmo_features_{}.npy'.format(i))
            # X_Train.extend(np.load('X_train_elmo_features_{}.npy'.format(i)))
            X_Train = np.load('features/X_train_elmo_features_{}.npy'.format(i))
            print(X_Train.shape)
            X_Train = padSequencesKeras(np.array(X_Train), max_seq_len, toPadding)
            target = Y_train[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
            Xtarget_in = Xtrain[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
            Ytarget_out = Ytrain[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
            target_category_train = to_categorical(target, 42)
            print(X_Train.shape, target_category_train.shape)
            X_Train, Y_train_con = preparedata(X_Train, target_category_train, seq_length)
            print(X_Train.shape, Y_train_con.shape)
            context_model.load_weights(con_model_name)
            # context_model.train_on_batch(X_Train_con, Y_train_con)
            context_model.fit(X_Train, Y_train_con, epochs=1, batch_size=32, verbose=2,
                              callbacks=callbacks_con)  # , validation_split=0.13)
            # con_model_name.load_weights(model_name)
            loss, new_acc = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
            print('Context Score results:', new_acc)
            if old_acc < new_acc:
                context_model.save_weights(con_model_name)
                print('Weights are saved with {} % acc while old acc was {} %'.format(new_acc, old_acc))
                old_acc = new_acc
            X_Train = 0

            # model.fit(X_Train, target_category_train, verbose=2, callbacks=callbacks)
            # model.load_weights('weight_parameters')
            # evaluation = model.evaluate(X_Test, target_category_test, verbose=2)
            # print(evaluation, 'for i = {}'.format(str(i)))
            # new_acc = evaluation[1]
            # if old_acc < new_acc:
            #     model.save_weights('weight_parameters')
            #     print('Weights are saved with {}'.format(new_acc))
            #     old_acc = new_acc