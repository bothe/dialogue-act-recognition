import csv
import numpy as np


def ensemble_annotation(non_con_out, con_out, con_out_mean, utt_Speaker_train, utt_train_data,
                        utt_id_train_data, utt_Emotion_train_data, sentiment_labels=[],
                        meld_data=True, file_name='meld_emotion', write_final_csv=True):
    if write_final_csv:
        fieldnames = ['speaker', 'uttID', 'utterance', 'emotion', 'sentiment',
                      'non_con_out', 'con_out', 'con_out_mean', 'match', 'con_match',
                      'EDA', 'Cor_EDA', 'all_match']
        store_meld_in_csv = open('results/eda_' + file_name + '_dataset.csv', mode='w', newline='')
        writer = csv.DictWriter(store_meld_in_csv, fieldnames=fieldnames)
        writer.writeheader()

    none_matches = 0
    total_match = 0
    con_matches = 0
    any_two_matches = 0
    utt_info_rows = []
    for i in range(len(con_out)):
        match = "NotMatch"
        con_match = "NotConMatch"
        all_match = "NotMatch"
        matched_element, correct_element = '', 'CORRECT'
        if con_out_mean[i] == con_out[i] == non_con_out[i]:
            all_match = "AllMatch"
            matched_element = con_out[i]
            correct_element = con_out[i]
            total_match += 1
        elif con_out_mean[i] == con_out[i]:
            con_match = "ConMatch"
            matched_element = con_out[i]
            correct_element = con_out[i]
            con_matches += 1
        elif con_out[i] == non_con_out[i] or con_out_mean[i] == non_con_out[i]:
            any_two_matches += 1
            match = "Match"
            if con_out[i] == non_con_out[i]:
                matched_element = con_out[i]
                correct_element = con_out[i]
            elif con_out_mean[i] == non_con_out[i]:
                matched_element = con_out_mean[i]
                correct_element = con_out_mean[i]
        else:
            matched_element = con_out[i]
            none_matches += 1

        if meld_data:
            sentiment = sentiment_labels[i]
        else:
            sentiment = 'sentiment'

        utt_info_row = {'speaker': utt_Speaker_train[i].encode("utf-8"), 'uttID': utt_id_train_data[i],
                        'utterance': utt_train_data[i].encode("utf-8"),
                        'emotion': utt_Emotion_train_data[i], 'sentiment': sentiment,
                        'non_con_out': str(non_con_out[i]), 'con_out': str(con_out[i]),
                        'con_out_mean': str(con_out_mean[i]), 'EDA': matched_element, 'Cor_EDA': correct_element,
                        'match': match, 'con_match': con_match, 'all_match': all_match}

        if write_final_csv:
            writer.writerow(utt_info_row)
        utt_info_rows.append(utt_info_row)

    print("Matches in all lists(3): {}% and in context lists(2): {}%, any two matches: {}%, None matched: {}%".format(
        round((total_match / (i + 1)) * 100, 2), round((con_matches / (i + 1)) * 100, 2),
        round((any_two_matches / (i + 1)) * 100, 2), round((none_matches / (i + 1)) * 100, 2)))
    return utt_info_rows


def convert_predictions_to_indices(eda1, eda2, eda3, eda4, eda5, tags):
    def return_indices(con_out_strs):
        con_out_nums = []
        for item in con_out_strs:
            if item == 'fo_o':
                item = 'fo_o_fw_"_by_bc'
            con_out_nums.append(list(tags).index(item))
        return np.array(con_out_nums)

    eda1 = return_indices(eda1)
    eda3 = return_indices(eda3)
    eda2 = return_indices(eda2)
    eda4 = return_indices(eda4)
    eda5 = return_indices(eda5)
    nominal = False
    if nominal:
        return np.reshape(np.concatenate((eda1, eda3, eda2, eda4, eda5)), (5, len(eda1))).transpose()
    else:
        return np.array([eda1, eda3, eda2, eda4, eda5])


def ensemble_eda_annotation(eda1, eda2, eda3, eda4, eda5,
                            eda1_conf, eda2_conf, eda3_conf, eda4_conf, eda5_conf,
                            utt_speaker, utterances, utt_id, utt_emotion,
                            sentiment_labels=[], meld_data=True,
                            file_name='meld_emotion', write_final_csv=True, write_utterances=True):
    if write_final_csv:
        fieldnames = ['speaker', 'utt_id', 'utterance', 'emotion', 'sentiment',
                      'eda1', 'eda2', 'eda3', 'eda4', 'eda5', 'EDA',
                      'all_match', 'con_match', 'match']

        store_meld_in_csv = open('annotated_data/eda_' + file_name + '_dataset.csv', mode='w', newline='')
        writer = csv.DictWriter(store_meld_in_csv, fieldnames=fieldnames)
        writer.writeheader()

    none_matches = 0
    total_match = 0
    con_matches = 0
    based_on_confs = 0
    utt_info_rows = []
    for i in range(len(eda1)):
        match = "NotMatch"
        con_match = "NotConMatch"
        all_match = "AllNotMatch"
        matched_element = 'xx'
        # All labels
        if eda5[i] == eda4[i] == eda3[i] == eda2[i] == eda1[i]:
            all_match = "AllMatch"
            matched_element = eda5[i]
            total_match += 1
        # Context labels
        elif eda5[i] == eda4[i] == eda3[i]:
            con_match = "ConMatch"
            matched_element = eda5[i]
            con_matches += 1
        elif eda5[i] == eda3[i] and eda5[i] in [eda1[i], eda2[i]]:
            con_match = "ConMatch_eda5_eda3"
            matched_element = eda5[i]
            con_matches += 1
        elif eda5[i] == eda4[i] and eda5[i] in [eda1[i], eda2[i]]:
            con_match = "ConMatch_eda5_eda4"
            matched_element = eda4[i]
            con_matches += 1
        elif eda4[i] == eda3[i] and eda4[i] in [eda1[i], eda2[i]]:
            con_match = "ConMatch_eda4_eda3"
            matched_element = eda4[i]
            con_matches += 1

        # None of above resulted any label, rank the confidence values
        else:
            temp_edas = [eda1[i], eda2[i], eda3[i], eda4[i], eda5[i]]
            temp_edas_conf = np.array([eda1_conf[i], eda2_conf[i], eda3_conf[i], eda4_conf[i], eda5_conf[i]])
            sorted_temp_edas_conf = np.flip(np.argsort(temp_edas_conf))
            opt_eda1 = temp_edas[sorted_temp_edas_conf[0]]
            opt_eda2 = temp_edas[sorted_temp_edas_conf[1]]
            opt_eda3 = temp_edas[sorted_temp_edas_conf[2]]
            opt_eda4 = temp_edas[sorted_temp_edas_conf[3]]
            if opt_eda1 == opt_eda2 == opt_eda3:
                matched_element = opt_eda1  # first order
                match = "ConfMatch" + 'First123'
                based_on_confs += 1
            elif opt_eda1 == opt_eda2 == opt_eda4:
                matched_element = opt_eda1  # second order
                match = "ConfMatch" + 'Second124'
                based_on_confs += 1
            elif opt_eda1 == opt_eda2:
                matched_element = opt_eda1  # second order
                match = "ConfMatch" + 'OnlyTwo12'
                based_on_confs += 1

        if meld_data:
            sentiment = sentiment_labels[i]
        else:
            sentiment = 'sentiment'

        # We would do this for IEMOCAP
        if write_utterances:
            utterance = utterances[i]
        else:
            utterance = ''

        utt_info_row = {'speaker': utt_speaker[i].encode("utf-8"), 'utt_id': utt_id[i],
                        'utterance': utterance.encode("utf-8"),
                        'emotion': utt_emotion[i], 'sentiment': sentiment,

                        'eda1': str(eda1[i]), 'eda2': str(eda2[i]), 'eda3': str(eda3[i]),
                        'eda4': str(eda4[i]), 'eda5': str(eda5[i]), 'EDA': matched_element,
                        'all_match': all_match, 'con_match': con_match, 'match': match}

        if write_final_csv:
            writer.writerow(utt_info_row)
        if matched_element == 'xx':
            none_matches+=1
        utt_info_rows.append(utt_info_row)

    print(
        "Matches in all: {}%, in context: {}%, based on confidence rank: {}%, and none matched: {}%".format(
            round((total_match / len(eda1)) * 100, 2), round((con_matches / len(eda1)) * 100, 2),
            round((based_on_confs / len(eda1)) * 100, 2), round((none_matches / len(eda1)) * 100, 2)))
    return utt_info_rows
