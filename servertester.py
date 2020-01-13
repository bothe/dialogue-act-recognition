import requests

speaker_ids = ["A", "B", "A", "B", "A", "B"]
utterances = ["I don't know, ", "Where did you go?", "What?", " Where did you go?", "I went to University.", "Uh-huh."]
utt_ids = ["1", "2", " 3", "4", "5", "6"]
emotions = ["neutral", "surprise", "surprise", "angry", "frustration", "neutral"]

text = "$$$$".join(speaker_ids) + "??????" + "$$$$".join(utterances) + "??????" + "$$$$".join(utt_ids) + "??????" + "$$$$".join(emotions)

link = "http://0.0.0.0:4004/"
link = "http://d55da20d.eu.ngrok.io/"
results = requests.post(link + 'predict_das', json={"text": text}).json()['result']

result_items = results.split('??????')
f_kappa_score_text = result_items[0]
k_alpha_score_text = result_items[1]
overall_data_assessment = result_items[2]

speaker_id, utt_id, utterance, emotion = [], [], [], []
eda1, eda2, eda3, eda4, eda5, EDA = [], [], [], [], [], []
all_match, con_match, match = [], [], []

for item in result_items[3:]:
    elements = item.split('$$$$')
    speaker_id.append(elements[0]), utt_id.append(elements[1]), utterance.append(elements[2]), emotion.append(elements[3])
    eda1.append(elements[4]), eda2.append(elements[5]), eda3.append(elements[6]), eda4.append(elements[7])
    eda5.append(elements[8]), EDA.append(elements[9])
    all_match.append(elements[10]), con_match.append(elements[11]), match.append(elements[12])
# print(results)
