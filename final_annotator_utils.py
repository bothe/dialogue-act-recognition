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
            con_match = "ConMatch_eda5_eda3_strong"
            matched_element = eda5[i]
            con_matches += 1
        elif eda5[i] == eda4[i] and eda5[i] in [eda1[i], eda2[i]]:
            con_match = "ConMatch_eda5_eda4_strong"
            matched_element = eda4[i]
            con_matches += 1
        elif eda4[i] == eda3[i] and eda4[i] in [eda1[i], eda2[i]]:
            con_match = "ConMatch_eda4_eda3_strong"
            matched_element = eda4[i]
            con_matches += 1
        elif eda5[i] == eda3[i]:
            con_match = "ConMatch_eda5_eda3_con"
            matched_element = eda5[i]
            con_matches += 1
        elif eda5[i] == eda4[i]:
            con_match = "ConMatch_eda5_eda4_con"
            matched_element = eda4[i]
            con_matches += 1
        elif eda4[i] == eda3[i]:
            con_match = "ConMatch_eda4_eda3_con"
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
            none_matches += 1
        utt_info_rows.append(utt_info_row)

    print(
        "Matches in all: {}%, in context: {}%, based on confidence rank: {}%, and none matched: {}%".format(
            round((total_match / len(eda1)) * 100, 2), round((con_matches / len(eda1)) * 100, 2),
            round((based_on_confs / len(eda1)) * 100, 2), round((none_matches / len(eda1)) * 100, 2)))
    return utt_info_rows


def fleiss_kappa(ratings, n):
    """
    Computes the Fleiss' kappa measure for assessing the reliability of
    agreement between a fixed number n of raters when assigning categorical
    ratings to a number of items.

    Args:
        ratings: a list of (item, category)-ratings
        n: number of raters
        k: number of categories
    Returns:
        the Fleiss' kappa score

    See also:
        http://en.wikipedia.org/wiki/Fleiss'_kappa
    """
    items = set()
    categories = set()
    n_ij = {}

    for i, c in ratings:
        items.add(i)
        categories.add(c)
        n_ij[(i, c)] = n_ij.get((i, c), 0) + 1

    N = len(items)

    p_j = dict(((c, sum(n_ij.get((i, c), 0) for i in items) / (1.0 * n * N)) for c in categories))
    P_i = dict(((i, (sum(n_ij.get((i, c), 0) ** 2 for c in categories) - n) / (n * (n - 1.0))) for i in items))

    P_bar = sum(P_i.values()) / (1.0 * N)
    P_e_bar = sum(value ** 2 for value in p_j.values())

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa


'''
Created on Aug 1, 2016
@author: skarumbaiah
Computes Fleiss' Kappa 
Joseph L. Fleiss, Measuring Nominal Scale Agreement Among Many Raters, 1971.
'''


def checkInput(rate, n):
    """
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(int(rate[i][j]), int) for i in range(N) for j in range(k)), "Element not integer"
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"


def fleissKappa(rate, n):
    """
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    checkInput(rate, n)

    # mean of the extent to which raters agree for the ith subject
    PA = sum([(sum([i ** 2 for i in row]) - n) / (n * (n - 1)) for row in rate]) / N
    print("PA = ", PA)

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j ** 2 for j in [sum([rows[i] for rows in rate]) / (N * n) for i in range(k)]])
    print("PE =", PE)

    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)

    return kappa


def kappa_data(reliability_data):
    global final_list, noted_samples
    final_list = []
    noted_samples = 0
    for item in reliability_data.transpose():
        temp_list = [0] * (reliability_data.max() + 1)
        for i in item:
            for j in range(len(temp_list)):
                if i == j + 1:
                    temp_list[j] = temp_list[j] + 1
        if sum(temp_list) == 5:  # only a few (around 200 out of 10000) are excluded
            final_list.append(temp_list)
            noted_samples += 1
    return final_list, noted_samples