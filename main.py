from collections import Counter
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import pickle

from models import model_attention_applied_after_bilstm, \
    model_attention_applied_after_bisrnn, dummyModel
from utils import *

trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
# from elmo_utils import get_elmo_tokens
# x_train = get_elmo_tokens(Xtrain)
# x_test = get_elmo_tokens(Xtest)
# pickle.dump( x_test, open( "x_test_tokens.p", "wb" ) )
# pickle.dump( x_train, open( "x_train_tokens.p", "wb" ) )

x_test = pickle.load(open("features/x_test_tokens.p", "rb"))
x_train = pickle.load(open("features/x_train_tokens.p", "rb"))

# toPadding = X_Train[27229][0]
toPadding = np.load('features/pad_a_token.npy')
X_Test = np.load('features/X_test_elmo_features.npy')
X_Test = padSequences(X_Test, toPadding)

tag, num = [], []
for i, j in enumerate((Counter(Ztrain)).keys()):
    tag.append(j)
    num.append(i)
Y_train = []
for i in Ztrain:
    Y_train.append(tag.index(i))
Y_test = []
for i in Ztest:
    Y_test.append(tag.index(i))

target_category_test = to_categorical(Y_test, 42)

SINGLE_ATTENTION_VECTOR = False
model = model_attention_applied_after_bilstm(20, 1024, 42, SINGLE_ATTENTION_VECTOR)
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

m = model_attention_applied_after_bisrnn(TIME_STEPS, INPUT_DIM, NUM_CLASSES, SINGLE_ATTENTION_VECTOR)
m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(m.summary())
# model_name = 'SLSTMSwDA'+str(seq_length)
model_name = 'params/BiSRNNSwDAPhonemes' + str(seq_length)
callbacks_con = [EarlyStopping(patience=3), ModelCheckpoint(filepath=model_name, save_best_only=True)]

train = True

if train == False:
    evaluation = model.evaluate(X_Test, target_category_test, verbose=2)
    print(evaluation)
else:
    old_acc = 0
    for iteration in range(10):
        print('Iteration number {}'.format(str(iteration + 1)))
        for i in range(8):
            i += 1
            print('X_train_elmo_features_{}.npy'.format(i))
            # X_Train.extend(np.load('X_train_elmo_features_{}.npy'.format(i)))
            X_Train = np.load('features/X_train_elmo_features_{}.npy'.format(i))
            X_Train = padSequences(np.array(X_Train), toPadding)
            target = Y_train[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
            Xtarget_in = Xtrain[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
            Ytarget_out = Ytrain[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
            target_category_train = to_categorical(target, 42)
            print(X_Train.shape, target_category_train.shape)
            X_Train_con, Y_train_con = preparedata(X_Train, target_category_train, seq_length)
            print(X_Train_con.shape, Y_train_con.shape)

            m.fit(X_Train_con, Y_train_con, epochs=1, batch_size=32, verbose=2,
                  callbacks=callbacks_con)  # , validation_split=0.13)
            m.load_weights(model_name)
            loss, new_acc = m.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
            print('Context Score results:', new_acc)
            if old_acc < new_acc:
                model.save_weights('params/weight_parameters')
                print('Weights are saved with {}'.format(new_acc))
                old_acc = new_acc

            # model.fit(X_Train, target_category_train, verbose=2, callbacks=callbacks)
            # model.load_weights('weight_parameters')
            evaluation = model.evaluate(X_Test, target_category_test, verbose=2)
            print(evaluation, 'for i = {}'.format(str(i)))
            # new_acc = evaluation[1]
            # if old_acc < new_acc:
            #     model.save_weights('weight_parameters')
            #     print('Weights are saved with {}'.format(new_acc))
            #     old_acc = new_acc
