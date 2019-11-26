from mocap_utils import *
import csv
import pandas as pd


def get_mocap_data(write=False, read_from_csv=False, csv_file_name="IEMOCAP/mocap_dataset.csv"):

    """
    get_directory_structure("IEMOCAP") - Sorts differently on Windows and Ubuntu,
    hence better to save csv after we read all data, and use that one latter
    :param write: write the csv file
    :param csv_file_name: provide correct path
    :param read_from_csv: read from the written file
    :return: lists of utterances, emo_dialogues, emo_evo, v, a, d, speaker_id
    """

    utterances = []
    emo_dialogues = []
    emo_evo = []
    v, a, d = [], [], []
    speaker_id = []
    file_paths = []
    transcriptions = {}
    emotions = {}
    fieldnames = ['speaker_id', 'utterance', 'emotion', 'v', 'a', 'd', 'emo_evo', 'start', 'end']

    if not read_from_csv:
        if write:
            store_mocap_in_csv = open(csv_file_name, mode='w', newline='')
            writer = csv.DictWriter(store_mocap_in_csv, fieldnames=fieldnames)
            writer.writeheader()

        path_tree = get_directory_structure("IEMOCAP")
        sessions = ['S1', 'S2', 'S3', 'S4', 'S5']
        for session in sessions:
            for text in path_tree['IEMOCAP'][session]["transcriptions"]:
                file_path_utt = "IEMOCAP/" + session + "/transcriptions/" + text
                file_path_emo = "IEMOCAP/" + session + "/EmoEvaluation/" + text
                utts = get_transcriptions(file_path_utt)
                emots = get_emotions(file_path_emo)
                transcriptions[session + '_' + text] = utts
                emotions[session + '_' + text] = emots
                for key in list(utts.keys()):
                    try:
                        emo_dialogues.append(emots[key]['emotion'])
                        emo_evo.append(emots[key]['emo_evo'])
                        v.append(emots[key]['v'])
                        a.append(emots[key]['a'])
                        d.append(emots[key]['d'])
                        utterances.append(utts[key])
                        speaker_id.append(key)
                        if write:
                            writer.writerow(
                                {'speaker_id': key, 'utterance': str(utts[key]), 'emotion': emots[key]['emotion'],
                                 'v': emots[key]['v'], 'a': emots[key]['a'], 'd': emots[key]['d'],
                                 'emo_evo': emots[key]['emo_evo'], 'start': emots[key]['start'], 'end': emots[key]['end']})

                    except KeyError:
                        pass

                file_paths.append(file_path_utt)
    else:
        df = pd.read_csv(csv_file_name)
        utterances = df['utterance'].tolist()
        emo_dialogues = df['emotion'].tolist()
        v, a, d = df['v'].tolist(), df['a'].tolist(), df['d'].tolist()
        speaker_id = df['speaker_id'].tolist()

    return utterances, emo_dialogues, emo_evo, v, a, d, speaker_id


# utterances, emo_dialogues, emo_evo, v, a, d, speaker_id = get_mocap_data(wirte=False)
# print('debug')
