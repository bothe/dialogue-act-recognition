import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from diswiz.main_context_predictor import *
from diswiz.utils_server import *
from diswiz.utils import prepare_input_data


def predict_das_diswiz(value):
    # value = value.split('\r\n')
    utts_s, DAname_s, confs_s, higher_DA_class_s = [], [], [], []
    non_con_das, non_con_da_nums, con_das, con_da_nums = [], [], ['None', 'None'], []
    for it_value in value:
        x_seq = pad_sequences(tokenizer.texts_to_sequences([it_value]), maxlen=MAX_SEQUENCE_LENGTH)
        predictions = non_con_model.predict(x_seq, verbose=2)
        it_value, classes, DAnames, confs = prepare_output(predictions, tag, it_value)
        # print('Text:=>', it_value)
        # print('DAs: =>', classes, DAnames, confs)
        non_con_das.append(tag[predictions[0].argmax()])
        non_con_da_nums.append(predictions[0].argmax())
        utts_s.append(it_value)
        DAname_s.append(DAnames[0:3])
        confs_s.append(confs[0:3])
        higher_DA_class_s.append(str(classes))

    if len(value) >= seq_len:
        x_con_seq = pad_sequences(tokenizer.texts_to_sequences(value), maxlen=MAX_SEQUENCE_LENGTH)
        x_con_seq = prepare_input_data(x_con_seq, seq_len)
        Con_DANames, Con_confs_s, Con_higher_DA_class_s = [], [], []
        for iterr in x_con_seq:
            predictions = context_model.predict(np.array([iterr]), verbose=2)
            it_value, ConClasses, ConDAnames, ConConfs = prepare_output(predictions, tag, iterr)
            con_das.append(tag[predictions[0].argmax()])
            con_da_nums.append(predictions[0].argmax())
            # print('Context DAs: =>', ConClasses, ConDAnames, ConConfs)
            Con_DANames.append(ConDAnames[0:3])
            Con_confs_s.append(ConConfs[0:3])
            Con_higher_DA_class_s.append(ConClasses)
    else:
        literal = ['NotEnoughContext', 'NotEnoughContext']
        Con_DANames = literal, Con_confs_s = literal, Con_higher_DA_class_s = literal

    res = {'most_probable_DA_names': DAname_s, 'most_probable_DA_confs': confs_s,
           'higher_DA_class': higher_DA_class_s, 'utts': utts_s,
           'Con_DANames': Con_DANames, 'Con_confs_s': Con_confs_s, 'Con_higher_DA_class_s': Con_higher_DA_class_s}

    return con_das, non_con_das, con_da_nums, non_con_da_nums
