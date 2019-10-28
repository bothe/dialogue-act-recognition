from keras.layers import Embedding, Lambda, Activation, Dropout
from keras.layers import Dense, Input, Flatten, RepeatVector, Permute
from keras.layers import merge, GRU, LSTM, Bidirectional, TimeDistributed
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    ''' https://github.com/philipperemy/keras-visualize-activations '''
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False
    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]
    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations


def utt_model(word_index, EMBEDDING_DIM, classes, MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2, W_reg=0.01):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(GRU(nodes, dropout=0.2, recurrent_dropout=0.2, name='internal_rnn'))
    model.add(Dense(classes, activation='softmax'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def utt_context_model(word_index, EMBEDDING_DIM, classes, MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2, W_reg=0.01):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(GRU(nodes, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(classes, activation='softmax'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def att_model(word_index, EMBEDDING_DIM, classes, MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2, W_reg=0.01):
    sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(len(word_index) + 1,
                                   EMBEDDING_DIM,
                                   input_length=MAX_SEQUENCE_LENGTH,
                                   trainable=True)(sentence_input)
    lstm_sentence = Bidirectional(GRU(nodes, return_sequences=True,
                                      recurrent_dropout=dropout, W_regularizer=l2(W_reg)))(embedded_sequences)
    dense_sentence = Dense(nodes, activation='tanh')(lstm_sentence)
    # Attention Layer
    attention = Dense(1, activation='relu')(lstm_sentence)  # try diff act
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)  # try different activations
    attention = RepeatVector(nodes)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = merge([dense_sentence, attention], mode='mul')
    sentence_attention = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(nodes,))(sent_representation)
    preds = Dense(classes, activation='softmax')(sentence_attention)
    model = Model(sentence_input, preds)
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    # print("model fitting LSTM")
    print(model.summary())
    return model


def dummyModel(seq_len, word_index, EMBEDDING_DIM, classes, MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2, W_reg=0.01):
    # Encode each timestep
    in_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedded_sentence = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                                  trainable=True)(in_sentence)
    lstm_sentence = LSTM(nodes)(embedded_sentence)
    encoded_model = Model(in_sentence, lstm_sentence)

    sequence_input = Input(shape=(seq_len, MAX_SEQUENCE_LENGTH), dtype='int64')
    seq_encoded = TimeDistributed(encoded_model)(sequence_input)
    seq_encoded = Dropout(dropout)(seq_encoded)

    # Encode entire sentence
    seq_encoded = LSTM(nodes)(seq_encoded)

    # Prediction
    prediction = Dense(classes, activation='softmax')(seq_encoded)
    model = Model(sequence_input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # print(model.summary())
    return model


def contex_gru(seq_len, word_index, EMBEDDING_DIM, classes, MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2, W_reg=0.01):
    # Encode each timestep
    in_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedded_sentence = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                                  trainable=True)(in_sentence)
    lstm_sentence = Bidirectional(GRU(nodes))(embedded_sentence)
    encoded_model = Model(in_sentence, lstm_sentence)

    sequence_input = Input(shape=(seq_len, MAX_SEQUENCE_LENGTH), dtype='int64')
    seq_encoded = TimeDistributed(encoded_model)(sequence_input)
    seq_encoded = Dropout(dropout)(seq_encoded)

    # Encode entire sentence
    seq_encoded = Bidirectional(GRU(nodes))(seq_encoded)

    # Prediction
    prediction = Dense(classes, activation='softmax')(seq_encoded)
    model = Model(sequence_input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # print(model.summary())
    return model


def contex_guru(encoded_model, seq_len, word_index, EMBEDDING_DIM, classes, MAX_SEQUENCE_LENGTH, nodes=128, dropout=0.2,
                W_reg=0.01):
    # Encode each timestep
    in_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    # embedded_sentence = Embedding(len(word_index) + 1, EMBEDDING_DIM,
    #                              trainable=True)(in_sentence)
    # lstm_sentence = Bidirectional(GRU(nodes))(embedded_sentence)
    # encoded_model = Model(in_sentence, lstm_sentence)

    intermediate_layer_model = Model(inputs=encoded_model.input,
                                     outputs=encoded_model.get_layer('internal_rnn').output)
    # encoded_model1 = intermediate_layer_model.predict(in_sentence)

    sequence_input = Input(shape=(seq_len, MAX_SEQUENCE_LENGTH), dtype='int64')
    seq_encoded = TimeDistributed(intermediate_layer_model)(sequence_input)
    seq_encoded = Dropout(dropout)(seq_encoded)

    # Encode entire sentence
    seq_encoded = (LSTM(nodes))(seq_encoded)

    # Prediction
    prediction = Dense(classes, activation='softmax')(seq_encoded)
    model = Model(sequence_input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # print(model.summary())
    return model


# dummyModel(20000, 50)
# att_model(np.random.randint(0,100, size=100), 50, 10, 20, nodes=128, dropout = 0.2, W_reg = 0.01)
