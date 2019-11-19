import csv


def ensemble_annotation(non_con_out, con_out, con_out_mean, utt_Speaker_train, utt_train_data,
                        utt_id_train_data, utt_Emotion_train_data, sentiment_labels=[],
                        meld_data=True, file_name='meld_emotion', write_final_csv=True):
    if write_final_csv:
        fieldnames = ['speaker', 'uttID', 'utterance', 'emotion', 'sentiment',
                      'non_con_out', 'con_out', 'con_out_mean', 'match', 'con_match', 'pre-final-DA', 'all_match']
        store_meld_in_csv = open('results/eda_' + file_name + '_dataset.csv', mode='w', newline='')
        writer = csv.DictWriter(store_meld_in_csv, fieldnames=fieldnames)
        writer.writeheader()

    total_match = 0
    con_matches = 0
    any_two_matches = 0
    utt_info_rows = []
    for i in range(len(con_out)):
        match = "NotMatch"
        con_match = "NotConMatch"
        all_match = "NotMatch"
        matched_element = ''
        if con_out_mean[i] == con_out[i] == non_con_out[i]:
            all_match = "AllMatch"
            total_match += 1
        if con_out_mean[i] == con_out[i]:
            con_match = "ConMatch"
            con_matches += 1
        if con_out[i] == non_con_out[i] or con_out_mean[i] == con_out[i] or con_out_mean[i] == non_con_out[i]:
            any_two_matches += 1
            match = "Match"
            if con_out[i] == non_con_out[i]:
                matched_element = con_out[i]
            elif con_out_mean[i] == con_out[i]:
                matched_element = con_out[i]
            elif con_out_mean[i] == non_con_out[i]:
                matched_element = con_out_mean[i]

            if meld_data:
                sentiment = sentiment_labels[i]
            else:
                sentiment = 'sentiment'

            utt_info_row = {'speaker': utt_Speaker_train[i].encode("utf-8"),
                            'uttID': utt_id_train_data[i],
                            'utterance': utt_train_data[i].encode("utf-8"),
                            'emotion': utt_Emotion_train_data[i], 'sentiment': sentiment,
                            'non_con_out': str(non_con_out[i]), 'con_out': str(con_out[i]),
                            'con_out_mean': str(con_out_mean[i]), 'pre-final-DA': matched_element,
                            'match': match, 'con_match': con_match, 'all_match': all_match}

        if write_final_csv:
            writer.writerow(utt_info_row)
        utt_info_rows.append(utt_info_row)

    print("Matches in all lists(3): {}% and in context lists(2): {}%, any two matches: {}%".format(
        round((total_match / (i + 1)) * 100, 2), round((con_matches / (i + 1)) * 100, 2),
        round((any_two_matches / (i + 1)) * 100, 2)))
    return utt_info_rows
