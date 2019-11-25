from keras.utils import to_categorical
import pickle, os
import requests

# TODO: need statements "if train" similar to main_swda_elmo_mean.py

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
print("Test results for non-context model accuracy: {}".format(evaluation[1]))

# Preparing test data for contextual training with Seq_length
seq_length = 3
X_test_con, Y_test_con = prepare_data(X_Test, target_category_test, seq_length)

# CONTEXT MODEL
context_model_elmo = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(tags))
con_model_name = 'params/context_model_att_{}'.format(seq_length)
context_model_elmo.load_weights(con_model_name)
loss, old_acc = context_model_elmo.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
print('Context Score result:', old_acc)

# TOP CONTEXT MODEL
top_context_model = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(tags), train_with_mean=True)
top_con_model_name = 'params/top_context_model_att_{}'.format(seq_length)
if os.path.exists(top_con_model_name):
    top_context_model.load_weights(top_con_model_name)
    loss, old_acc = top_context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
    print('Top Context Score result:', old_acc)


def predict_classes_for_elmo(x, predict_from_text=False, link_online=False):
    ''' Predicting from text takes 'x' as a list of utterances and
    will require to have ELMo emb server running at port 4004 or online hosting service. '''
    if predict_from_text:
        if link_online:
            link = "https://d55da20d.eu.ngrok.io/"
        else:
            link = "http://0.0.0.0:4004/"
        x = string_to_floats(
            requests.post(link + 'elmo_embed_words',
                          json={"text": x}).json()['result'])

    x = padSequencesKeras(x, max_seq_len, toPadding)

    non_con_predictions = model.predict(x)
    non_con_out = []
    non_con_out_confs = []
    for item in non_con_predictions:
        non_con_out.append(tags[item.argmax()])
        non_con_out_confs.append(item[item.argmax()])

    if len(x) > seq_length:
        x = prepare_data(x, [], seq_length, with_y=False)
        con_predictions = context_model_elmo.predict(x)
        top_con_predictions = top_context_model.predict(x)

        # as context based model starts from third utterance
        # we are taking first two DAs from non-context model
        # in the end, it will be checked with their conf values
        con_out = [non_con_out[0], non_con_out[1]]
        con_out_confs = [non_con_out_confs[0], non_con_out_confs[1]]
        top_con_out = [non_con_out[0], non_con_out[1]]
        top_con_out_confs = [non_con_out_confs[0], non_con_out_confs[1]]
        for item in con_predictions:
            con_out.append(tags[np.argmax(item)])
            con_out_confs.append(item[item.argmax()])
        for item in top_con_predictions:
            top_context_model.append(tags[np.argmax(item)])
            top_context_model.append(item[item.argmax()])
    else:
        con_out = ['None', 'None']
        con_out_confs = [0.0, 0.0]
        top_con_out = ['None', 'None']
        top_con_out_confs = [0.0, 0.0]
    return non_con_out, con_out, non_con_out_confs, con_out_confs, top_con_out, top_con_out_confs

# manual tests
# print(predict_classes_for_elmo(
#     "I don't know,  \r\n Where did you go? \r\n  What? \r\n  "
#     "Where did you go? \r\n I went to University. \r\n Uh-huh.",
#     predict_from_text=True, link_online=False))
# print(predict_classes("\r\n".join(Xtest[0:100]),
#     predict_from_text=True, link_online=False))
# print(Ytest[0:100])
