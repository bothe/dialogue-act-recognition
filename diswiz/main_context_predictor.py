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
                           param_file_nc_new, param_file_con_new,
                           train=False, epochs=5):
    sid_train, x_train_txt, y_train_txt, z_train_txt = read_files(train_data)
    sid_test, x_test_txt, y_test_txt, z_test_txt = read_files(test_data)
    tag, num, y_test, y_train = prepare_targets(z_train_txt, z_test_txt)
    y_test = to_categorical(y_test, 42)
    y_train = to_categorical(y_train, 42)
    texts = x_train_txt + x_test_txt
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)  # , filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t\n',split='')
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Non-context test data
    x_test = tokenizer.texts_to_sequences(x_test_txt)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of input tensor:', x_test.shape)
    print('Shape of label tensor:', y_test.shape)

    # Non-context model with params
    non_con_model = utt_model(word_index, EMBEDDING_DIM, y_test.shape[1], MAX_SEQUENCE_LENGTH,
                              nodes=128, dropout=0.2, W_reg=0.01)
    non_con_model.summary()
    non_con_model.load_weights(param_file_nc)

    # Context test data
    seq_len = 3
    x_test_con, y_test_con = preparedata(x_test, y_test, seq_len)

    # Context model
    context_model = dummyModel(seq_len, word_index, EMBEDDING_DIM, y_test_con.shape[1], MAX_SEQUENCE_LENGTH,
                               nodes=128, dropout=0.2, W_reg=0.01)
    context_model.summary()
    context_model.load_weights(param_file_con)
    if not train:
        # Non-context model evaluation
        non_con_model_acc = non_con_model.evaluate(x_test, y_test, verbose=2)
        print('Non-con-model accuracy', non_con_model_acc)

        # Context model evaluation
        con_model_acc = context_model.evaluate(x_test_con, y_test_con, verbose=2)
        print('Non-con-model accuracy', con_model_acc)
    else:
        x_train = tokenizer.texts_to_sequences(x_train_txt)
        x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of input tensor:', x_train.shape)
        print('Shape of label tensor:', y_train.shape)

        # Train non-context model
        check_pointer = ModelCheckpoint(param_file_nc_new, save_best_only=True)
        non_con_model.fit(x_train, y_train, validation_split=0.15, epochs=epochs,
                          callbacks=[check_pointer], verbose=2)
        non_con_model.load_weights(param_file_nc_new)
        print(non_con_model.evaluate(x_test, y_test))

        x_train_con, y_train_con = preparedata(x_train, y_train, seq_len)
        print('Shape of input tensor:', x_train_con.shape)
        print('Shape of label tensor:', y_train_con.shape)
        # Train context model
        check_pointer = ModelCheckpoint(param_file_con_new, save_best_only=True)
        context_model.fit(x_train_con, y_train_con, validation_split=0.15, epochs=epochs,
                          callbacks=[check_pointer], verbose=2)
        context_model.load_weights(param_file_con_new)
        print(context_model.evaluate(x_test_con, y_test_con))


print('debug')
