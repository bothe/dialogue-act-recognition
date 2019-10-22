from mocap_utils import *


path_tree = get_directory_structure("IEMOCAP")

sessions = ['S1', 'S2', 'S3', 'S4', 'S5']

list_dialogues = []
file_paths = []
transcriptions = {}
emotions = {}
for session in sessions:
    for text in path_tree['IEMOCAP'][session]["transcriptions"]:
        file_path_utt = "IEMOCAP/" + session + "/transcriptions/" + text
        file_path_emo = "IEMOCAP/" + session + "/EmoEvaluation/" + text
        utts = get_transcriptions(file_path_utt)
        emots = get_emotions(file_path_emo)
        transcriptions[session + '_' + text] = utts
        emotions[session + '_' + text] = emots
        list_dialogues.extend(utts)
        file_paths.append(file_path_utt)

print('debug')
