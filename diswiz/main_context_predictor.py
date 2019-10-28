from keras.utils import to_categorical
from diswiz.utils import read_files, preparedata, prepare_targets
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from diswiz.models import utt_model, dummyModel

# CONFIGS
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 50

SidTr, Xtrain, Ytrain, Ztrain = read_files('data/swda-actags_train_speaker.csv')
SidTest, Xtest, Ytest, Ztest = read_files('data/swda-actags_test_speaker.csv')
tag, num, Y_test, Y_train = prepare_targets(Ztrain, Ztest)
Y_test = to_categorical(Y_test, 42)
Y_train = to_categorical(Y_train, 42)
texts = Xtrain + Xtest
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)  # , filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t\n',split='')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_test = tokenizer.texts_to_sequences(Xtest)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of input tensor:', X_test.shape)
print('Shape of label tensor:', Y_test.shape)

# Model without context
param_file_nc = 'diswiz/params/params_non_context'
non_con_model = utt_model(word_index, EMBEDDING_DIM, Y_test.shape[1], MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2,
                          W_reg=0.01)
non_con_model.summary()
non_con_model.load_weights(param_file_nc)
non_con_model_acc = non_con_model.evaluate(X_test, Y_test, verbose=2)
print('Non-con-model accuracy', non_con_model_acc)

# Context data
seq_len = 3
X_test, Y_test = preparedata(X_test, Y_test, seq_len)

context_model = dummyModel(seq_len, word_index, EMBEDDING_DIM, Y_test.shape[1], MAX_SEQUENCE_LENGTH, nodes=128,
                           dropout=0.2, W_reg=0.01)
context_model.summary()
param_file_con = 'diswiz/params/params_context'
context_model.load_weights(param_file_con)
con_model_acc = context_model.evaluate(X_test, Y_test, verbose=2)
print('Non-con-model accuracy', con_model_acc)

print('debug')
