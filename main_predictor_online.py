from keras.utils import to_categorical
import pickle, os
import requests
# requests.post('http://0:4004/elmo_embed_words', json={"text":'is it?'}).json()
# requests.post('https://55898a32.eu.ngrok.io/elmo_embed_words', json={"text":'is it?\r\nokay got it.'}).json()

from models import model_attention_applied_after_bilstm, context_model_att
from utils import *
from utils_float_string import *

max_seq_len = 20
trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)
print(len(Xtest), len(Xtrain))
x_test = pickle.load(open("features/x_test_tokens.p", "rb"))
x_train = pickle.load(open("features/x_train_tokens.p", "rb"))
toPadding = np.load('features/pad_a_token.npy')

X_Test = np.load('features/X_test_elmo_features.npy', allow_pickle=True)
X_Test = padSequencesKeras(X_Test, max_seq_len, toPadding)
tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
target_category_test = to_categorical(Y_test, len(tags))

# NON-CONTEXT MODEL
model = model_attention_applied_after_bilstm(max_seq_len, X_Test.shape[2], len(tags))
model.load_weights('params/weight_parameters')
evaluation = model.evaluate(X_Test, target_category_test, verbose=2)
print("Test results for non-context model - accuracy: {}".format(evaluation[1]))

seq_length = 3  # Preparing data for contextual training with Seq_length
X_test_con, Y_test_con = prepare_data(X_Test, target_category_test, seq_length)

# CONTEXT MODEL
context_model = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(tags))
con_model_name = 'params/context_model_att_{}'.format(seq_length)
context_model.load_weights(con_model_name)
loss, old_acc = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
print('Context Score results:', old_acc)


def predict_classes(text_input, link_online=False):
    if link_online:
        link = "https://136c22af.eu.ngrok.io/"
    else:
        link = "http://0.0.0.0:4004/"
    x = string_to_floats(
        requests.post(link + 'elmo_embed_words',
                      json={"text": text_input}).json()['result'])
    x = padSequencesKeras(x, max_seq_len, toPadding)
    non_con_predictions = model.predict(x)
    non_con_out = []
    for item in non_con_predictions:
        non_con_out.append(tags[np.argmax(item)])

    if len(x) > seq_length:
        x = prepare_data(x, [], seq_length, with_y=False)
        con_predictions = context_model.predict(x)
        con_out = []
        for item in con_predictions:
            con_out.append(tags[np.argmax(item)])

    else:
        con_out = []

    return non_con_out, con_out


print(predict_classes(
    "I don't know,  \r\n Where did you go? \r\n  What? \r\n  "
    "Where did you go? \r\n I went to University. \r\n Uh-huh."))
print(predict_classes("\r\n".join(Xtest[0:100])))
print(Ytest[0:100])
