from keras.utils import to_categorical
import pickle, os
from models import model_attention_applied_after_bilstm, context_model_att
from utils import *


def main():
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
    model = model_attention_applied_after_bilstm(max_seq_len, X_Test.shape[2], len(tags))
    model.load_weights('params/weight_parameters')
    evaluation = model.evaluate(X_Test, target_category_test, verbose=2)
    print("Test results for non-context model - accuracy: {}".format(evaluation[1]))

    seq_length = 3  # Preparing data for contextual training with Seq_length
    X_test_con, Y_test_con = preparedata(X_Test, target_category_test, seq_length)

    # CONTEXT MODEL
    context_model = context_model_att(seq_length, max_seq_len, X_test_con.shape[3], len(tags))
    con_model_name = 'params/context_model_att_{}'.format(seq_length)
    context_model.load_weights(con_model_name)
    loss, old_acc = context_model.evaluate(X_test_con, Y_test_con, verbose=2, batch_size=32)
    print('Context Score results:', old_acc)



if __name__ == "__main__":
    main()
