from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from diswiz.utils import read_files, preparedata, prepare_targets
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from diswiz.models import utt_model, dummyModel

# CONFIGS
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 50


def diswiz_model_functions(train_data, test_data,
                           param_file_nc, param_file_con,
                           param_file_nc_new, param_file_con_new, train=False):
    SidTr, Xtrain, Ytrain, Ztrain = read_files(train_data)
    SidTest, Xtest, Ytest, Ztest = read_files(test_data)
    tag, num, Y_test, Y_train = prepare_targets(Ztrain, Ztest)
    Y_test = to_categorical(Y_test, 42)
    Y_train = to_categorical(Y_train, 42)
    texts = Xtrain + Xtest
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)  # , filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t\n',split='')
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Non-context test data
    X_test = tokenizer.texts_to_sequences(Xtest)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of input tensor:', X_test.shape)
    print('Shape of label tensor:', Y_test.shape)

    # Non-context model with params
    non_con_model = utt_model(word_index, EMBEDDING_DIM, Y_test.shape[1], MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2,
                              W_reg=0.01)
    non_con_model.summary()
    non_con_model.load_weights(param_file_nc)

    # Context test data
    seq_len = 3
    X_test_con, Y_test_con = preparedata(X_test, Y_test, seq_len)

    # Context model
    context_model = dummyModel(seq_len, word_index, EMBEDDING_DIM, Y_test_con.shape[1], MAX_SEQUENCE_LENGTH, nodes=128,
                               dropout=0.2, W_reg=0.01)
    context_model.summary()
    context_model.load_weights(param_file_con)
    if not train:
        # Non-context model evaluation
        non_con_model_acc = non_con_model.evaluate(X_test, Y_test, verbose=2)
        print('Non-con-model accuracy', non_con_model_acc)

        # Context model evaluation
        con_model_acc = context_model.evaluate(X_test_con, Y_test_con, verbose=2)
        print('Non-con-model accuracy', con_model_acc)
    else:
        X_train = tokenizer.texts_to_sequences(Xtrain)
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of input tensor:', X_train.shape)
        print('Shape of label tensor:', Y_train.shape)

        # Train non-context model
        check_pointer = ModelCheckpoint(param_file_nc_new, save_best_only=True)
        non_con_model.fit(X_train, Y_train, validation_split=0.1, epochs=10, callbacks=[check_pointer])
        non_con_model.load_weights(param_file_nc_new)
        non_con_model.evaluate(X_test, Y_test)


print('debug')
