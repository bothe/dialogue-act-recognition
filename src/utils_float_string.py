import numpy as np


def float_to_string(res):
    vectors = []
    for item in res:
        vector = []
        for element in item:
            vector.append(''.join(str(element.tolist())).replace('[', '').replace(']', ''))
        vectors.append(' $$ '.join(vector))
    return ' $$$$ '.join(vectors)


def string_to_floats(instrings):
    vectors = []
    substrings = instrings.split('$$$$')
    for item in substrings:
        vector = []
        substrings1 = item.split('$$')
        for elements in substrings1:
            embs = []
            str_elements = elements.split(',')
            for str_element in str_elements:
                embs.append(float(str_element))
            embs = np.array(embs)
            vector.append(embs)
        vector = np.array(vector)
        vectors.append(vector)
    return np.array(vectors)


def str_utils(text="", speaker_id=None, utterances=None, utt_id=None, emotion=None, mode='encode'):
    if emotion is None:
        emotion = []
    if utt_id is None:
        utt_id = []
    if utterances is None:
        utterances = []
    if speaker_id is None:
        speaker_id = []
    if mode == 'encode':
        text = "$$$$".join(speaker_id) + "?????" + "$$$$".join(utterances) + "?????" + \
               "$$$$".join(utt_id) + "?????" + "$$$$".join(emotion)
        return text
    else:
        encoded_text = text.split("?????")
        speaker_id = encoded_text[0].split("$$$$")
        utterances = encoded_text[1].split("$$$$")
        utt_id = encoded_text[2].split("$$$$")
        emotion = encoded_text[3].split("$$$$")
        return speaker_id, utterances, utt_id, emotion
