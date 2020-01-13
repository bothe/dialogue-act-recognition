import requests

speaker_ids = ["A", "B", "A", "B", "A", "B"]
utterances = ["I don't know, ", "Where did you go?", "What?", " Where did you go?", "I went to University.", "Uh-huh."]
utt_ids = ["1", "2", " 3", "4", "5", "6"]
emotions = ["neutral", "surprise", "surprise", "angry", "frustration", "neutral"]

text = "$$".join(speaker_ids) + "???" + "$$".join(utterances) + "???" + "$$".join(utt_ids) + "???" + "$$".join(emotions)

link = "http://0.0.0.0:4004/"
link = "http://d55da20d.eu.ngrok.io/"
results = requests.post(link + 'predict_das', json={"text": text}).json()

print(results)
