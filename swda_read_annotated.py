import os
import pandas as pd


import numpy as np
from sklearn.metrics import classification

from src.utils import read_files

elmo_feature_retrieval = False
predict_with_elmo = False
predict_with_elmo_mean = False

if os.path.exists('results/tags.npy'):
    tags = np.load('results/tags.npy')

# utterances, emotion, emo_evo, v, a, d, speaker_id, utt_id = get_swda_data(read_from_csv=True, write=True)
trainFile = 'data/swda-actags_train_speaker.csv'
testFile = 'data/swda-actags_test_speaker.csv'
toPadding = np.load('features/pad_a_token.npy', allow_pickle=True)
SidTr, Xtrain, Ytrain, Ztrain = read_files(trainFile)
SidTest, Xtest, Ytest, Ztest = read_files(testFile)

output_file = "annotated_data/eda_like_swda_final_annotation_dataset.csv"
df = pd.read_csv(output_file)
final_das = df['EDA'].tolist()

# Evaluation of context model predictions
print('Accuracy comparision between context-based predictions: {}'.format(
    classification.accuracy_score(final_das, Ztest)))

print(classification.classification_report(Ztest, final_das))
