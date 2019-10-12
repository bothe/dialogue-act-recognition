from keras.layers import Dense, SimpleRNN, LSTM, Input, Flatten, Bidirectional, GRU, TimeDistributed, Embedding
from keras.layers.merge import multiply, concatenate
from keras.layers.core import *
from keras.optimizers import Adam
from keras.engine import Model
from keras import backend as K


def dummyModel(seq_len, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, classes, nodes=128, dropout=0.2, W_reg=0.01):
    # Encode each time step
    in_sentence = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,), dtype='int64')
    # embedded_sentence = Embedding(len(word_index) + 1, EMBEDDING_DIM, trainable=True)(in_sentence)
    lstm_sentence = LSTM(nodes)(in_sentence)
    encoded_model = Model(in_sentence, lstm_sentence)

    # Model contextual time steps
    sequence_input = Input(shape=(seq_len, MAX_SEQUENCE_LENGTH), dtype='int64')
    seq_encoded = TimeDistributed(encoded_model)(sequence_input)
    seq_encoded = Dropout(dropout)(seq_encoded)
    # Encode entire sentence
    seq_encoded = LSTM(nodes)(seq_encoded)
    # Prediction
    prediction = Dense(classes, activation='softmax')(seq_encoded)

    model = Model(sequence_input, prediction)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


# dummyModel(3, 20, 1024, 42)

def attention_3d_block(inputs, TIME_STEPS, SINGLE_ATTENTION_VECTOR):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(K.int_shape(inputs)[2])
    a = Permute((2, 1))(inputs)
    # Reshape has no purpose except making the code more explicit and clear:
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    # when you apply a Dense layer, it applies to the last dimension of your tensor.
    # Permute is used to apply a Dense layer along the time axis (by default it's axis=1 in Keras)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def model_attention_applied_after_lstm(TIME_STEPS, INPUT_DIM, num_classes, SINGLE_ATTENTION_VECTOR):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 64
    # lstm_out      = (SimpleRNN(lstm_units, return_sequences=True))(inputs)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out, TIME_STEPS, SINGLE_ATTENTION_VECTOR)
    attention_mul = Flatten()(attention_mul)
    # inter_rep     = Dense(100)(attention_mul)
    output = Dense(num_classes, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    print(model.summary())
    return model


# model_attention_applied_after_lstm(3, 1024, 42, True)

def model_attention_applied_after_bilstm(TIME_STEPS, INPUT_DIM, num_classes, SINGLE_ATTENTION_VECTOR):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 64
    # lstm_out      = (SimpleRNN(lstm_units, return_sequences=True))(inputs)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    attention_mul = attention_3d_block(lstm_out, TIME_STEPS, SINGLE_ATTENTION_VECTOR)
    attention_mul = Flatten()(attention_mul)
    # inter_rep     = Dense(100)(attention_mul)
    output = Dense(num_classes, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    print(model.summary())
    return model


def model_attention_applied_after_bisrnn(TIME_STEPS, INPUT_DIM, num_classes, SINGLE_ATTENTION_VECTOR):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 64
    # lstm_out      = (SimpleRNN(lstm_units, return_sequences=True))(inputs)
    lstm_out = Bidirectional(SimpleRNN(lstm_units, return_sequences=True))(inputs)
    attention_mul = attention_3d_block(lstm_out, TIME_STEPS, SINGLE_ATTENTION_VECTOR)
    attention_mul = Flatten()(attention_mul)
    # inter_rep     = Dense(100)(attention_mul)
    output = Dense(num_classes, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    print(model.summary())
    return model
