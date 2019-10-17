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
