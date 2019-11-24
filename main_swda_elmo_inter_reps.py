from keras import Model
from keras.utils import to_categorical
import pickle, os
from models import model_attention_applied_after_bilstm
from utils import *


def main():
    max_seq_len = 20
    trainFile = 'data/swda-actags_train_speaker.csv'
    testFile = 'data/swda-actags_test_speaker.csv'
    SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
    SidTest, Xtest, Ytest, Ztest = read_files(testFile)
    print(len(Xtest), len(Xtrain))
    toPadding = np.load('features/pad_a_token.npy')

    X_Test = np.load('features/X_test_elmo_features.npy')

    X_Test = padSequencesKeras(X_Test, max_seq_len, toPadding)
    tags, num, Y_train, Y_test = categorize_raw_data(Ztrain, Ztest)
    target_category_test = to_categorical(Y_test, len(tags))

    # NON-CONTEXT MODEL
    non_con_model = model_attention_applied_after_bilstm(max_seq_len, X_Test.shape[2], len(tags))
    non_con_model.load_weights('params/weight_parameters')
    evaluation = non_con_model.evaluate(X_Test, target_category_test, verbose=2)
    print("Test results for non-context model - accuracy: {}".format(evaluation[1]))

    layer_name = 'flatten_attention'
    intermediate_layer_model = Model(inputs=non_con_model.input,
                                     outputs=non_con_model.get_layer(layer_name).output)

    intermediate_output_x_test = intermediate_layer_model.predict(X_Test)

    intermediate_output_x_train = []
    for i in range(8):
        i += 1
        print('X_train_elmo_features_{}.npy'.format(i))
        # X_Train.extend(np.load('X_train_elmo_features_{}.npy'.format(i)))
        X_Train = np.load('features/X_train_elmo_features_{}.npy'.format(i))
        print(X_Train.shape)
        X_Train = padSequencesKeras(np.array(X_Train), max_seq_len, toPadding)
        target = Y_train[(i - 1) * len(X_Train):(i - 1) * len(X_Train) + len(X_Train)]
        target_category_train = to_categorical(target, 42)
        print(X_Train.shape, target_category_train.shape)

        intermediate_output_x_train_temp = intermediate_layer_model.predict(X_Train)
        intermediate_output_x_train.extend(intermediate_output_x_train_temp)
        X_Train = 0
    np.save('features/swda_inter_reps/X_test_elmo_features_flatten_attention', intermediate_output_x_test)
    np.save('features/swda_inter_reps/X_train_elmo_features_flatten_attention', intermediate_output_x_train)




if __name__ == "__main__":
    main()
