from keras.layers import Dense, SimpleRNN, LSTM, Input, Flatten, Bidirectional, GRU, \
    TimeDistributed, Embedding, Conv1D, ConvLSTM2D, MaxPooling1D
from keras.layers.merge import multiply, concatenate
from keras.layers.core import *
from keras.optimizers import Adam
from keras.engine import Model
from keras import backend as K


def attention_3d_block(inputs, seq_length, single_attention_vector):
    # inputs.shape = (batch_size, seq_length, input_dim)
    input_dim = int(K.int_shape(inputs)[2])
    a = Permute((2, 1))(inputs)
    # Reshape has no purpose except making the code more explicit and clear:
    a = Reshape((input_dim, seq_length))(a)
    a = Dense(seq_length, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    # when you apply a Dense layer, it applies to the last dimension of your tensor.
    # Permute is used to apply a Dense layer along the time axis (by default it's axis=1 in Keras)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def context_model(seq_len, max_seq_length, emb_dim, classes, nodes=128, dropout=0.2,
                  single_attention_vector=False):
    # Encode each time step  # dummyModel(3, 20, 1024, 42)
    in_sentence = Input(shape=(max_seq_length, emb_dim,))  # , dtype='int64')
    # embedded_sentence = Embedding(len(word_index) + 1, emb_dim, trainable=True)(in_sentence)
    lstm_sentence = LSTM(nodes)(in_sentence)
    encoded_model = Model(in_sentence, lstm_sentence)
    encoded_model.summary()

    # Model contextual time steps
    sequence_input = Input(shape=(seq_len, max_seq_length, emb_dim))
    seq_encoded = TimeDistributed(encoded_model)(sequence_input)
    seq_encoded = Dropout(dropout)(seq_encoded)
    # Encode entire sentence
    seq_encoded = LSTM(nodes, return_sequences=True)(seq_encoded)
    # Apply attention layer
    attention_mul = attention_3d_block(seq_encoded, seq_len, single_attention_vector)
    attention_mul = Flatten()(attention_mul)
    # Prediction
    prediction = Dense(classes, activation='softmax')(attention_mul)

    model = Model(sequence_input, prediction)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def context_model_att(seq_len, max_seq_length, emb_dim, classes, nodes=128, dropout=0.2,
                      single_attention_vector=False):
    # Encode each time step  # dummyModel(3, 20, 1024, 42)
    in_sentence = Input(shape=(max_seq_length, emb_dim,))  # , dtype='int64')
    # embedded_sentence = Embedding(len(word_index) + 1, emb_dim, trainable=True)(in_sentence)
    lstm_sentence = LSTM(nodes, return_sequences=True)(in_sentence)
    # Apply attention layer
    attention_mul = attention_3d_block(lstm_sentence, max_seq_length, single_attention_vector)
    attention_mul = Flatten()(attention_mul)
    encoded_model = Model(in_sentence, attention_mul)
    encoded_model.summary()

    # Model contextual time steps
    sequence_input = Input(shape=(seq_len, max_seq_length, emb_dim))
    seq_encoded = TimeDistributed(encoded_model)(sequence_input)
    seq_encoded = Dropout(dropout)(seq_encoded)
    # Encode entire sentence
    seq_encoded = LSTM(nodes, return_sequences=True)(seq_encoded)
    # Apply attention layer
    attention_mul = attention_3d_block(seq_encoded, seq_len, single_attention_vector)
    attention_mul = Flatten(name='flatten_attention')(attention_mul)
    # Prediction
    prediction = Dense(classes, activation='softmax')(attention_mul)

    model = Model(sequence_input, prediction)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


# context_model(3, 20, 1024, 42)

def model_attention_applied_after_lstm(seq_length, emb_dim, num_classes, single_attention_vector=False):
    inputs = Input(shape=(seq_length, emb_dim,))
    lstm_units = 64
    # lstm_out      = (SimpleRNN(lstm_units, return_sequences=True))(inputs)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out, seq_length, single_attention_vector)
    attention_mul = Flatten()(attention_mul)
    # inter_rep     = Dense(100)(attention_mul)
    output = Dense(num_classes, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# model_attention_applied_after_lstm(3, 1024, 42, True)

def model_attention_applied_after_bilstm(seq_length, emb_dim, num_classes, single_attention_vector=False):
    inputs = Input(shape=(seq_length, emb_dim,))
    lstm_units = 64
    # lstm_out      = (SimpleRNN(lstm_units, return_sequences=True))(inputs)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    attention_mul = attention_3d_block(lstm_out, seq_length, single_attention_vector)
    attention_mul = Flatten(name='flatten_attention')(attention_mul)
    # inter_rep     = Dense(100)(attention_mul)
    output = Dense(num_classes, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def model_attention_applied_after_bisrnn(seq_length, emb_dim, num_classes, single_attention_vector=False):
    inputs = Input(shape=(seq_length, emb_dim,))
    lstm_units = 64
    # lstm_out      = (SimpleRNN(lstm_units, return_sequences=True))(inputs)
    lstm_out = Bidirectional(SimpleRNN(lstm_units, return_sequences=True))(inputs)
    attention_mul = attention_3d_block(lstm_out, seq_length, single_attention_vector)
    attention_mul = Flatten()(attention_mul)
    # inter_rep     = Dense(100)(attention_mul)
    output = Dense(num_classes, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def model_for_utterance_level(seq_length, emb_dim, num_classes, single_attention_vector=False):
    inputs = Input(shape=(seq_length, emb_dim,))
    lstm_units = 64
    # lstm_out      = (SimpleRNN(lstm_units, return_sequences=True))(inputs)
    lstm_out = Bidirectional(SimpleRNN(lstm_units, return_sequences=True))(inputs)
    attention_mul = attention_3d_block(lstm_out, seq_length, single_attention_vector)
    attention_mul = Flatten()(attention_mul)
    # inter_rep     = Dense(100)(attention_mul)
    output = Dense(num_classes, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def non_context_model_for_utterance_level(emb_dim, num_classes):
    inputs = Input(shape=(emb_dim,))
    units = 128
    reshape_features = Reshape((32, 32))(inputs)
    hidden_out = Conv1D(units, kernel_size=3, activation='relu')(reshape_features)
    hidden_out = Conv1D(units, kernel_size=3, activation='relu')(hidden_out)
    hidden_out_pooling = MaxPooling1D(pool_size=2)(hidden_out)
    inter_rep = Flatten()(hidden_out_pooling)
    inter_rep = Dense(100)(inter_rep)
    output = Dense(num_classes, activation='softmax')(inter_rep)
    model = Model(inputs=[inputs], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
