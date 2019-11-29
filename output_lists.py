from src.utils import *

SidTest, Xtest, Ytest, Ztest = read_files('data/swda-actags_test_speaker.csv')

non_con_pred = ['fo_o_fw_"_by_bc', 'qy', 'sv', 'sv', 'qo', 'sd', 'sd', 'sv', 'b', 'qy', 'nn', 'sd', 'sd', 'bk', 'sd',
                'qo', '%', 'sd', 'b', 'sd', 'sd', 'sd', 'bk', 'sd', 'sd', 'sd', 'b', 'sv', 'sd', 'sd', 'sd', 'sd', 'sd',
                'sd', 'sd', 'sd', 'b', 'sv', 'sd', 'sd', 'b', 'sd', 'b', 'b', '%', 'sd', '%', 'sv', 'sd', 'b', 'sd',
                'sv', 'sd', 'sd', 'bk', 'qy', 'b', 'sd', 'sd', 'x', 'b', 'sv', 'b', 'sv', 'b', 'sv', 'b', 'sd', 'sd',
                'b', 'b', 'sd', 'b', 'b', 'b', 'aa', 'sv', 'b', 'sv', 'qw', '%', 'sd', 'qy^d', 'b', 'b', 'sd', 'b',
                'b^m', 'b', 'sd', 'sd', '%', 'sd', 'b', 'nn', 'qy', 'sd', 'qy^d', 'sd', 'sd']
con_prediction = ['OO', 'OO', 'sv', 'sv', 'qo', 'sd', 'sd', 'sv', 'b', 'qy', 'nn', 'sd', 'sd', 'b', 'sd', 'qo', '%',
                  'sd', 'b',
                  'sd', 'sd', 'sd', 'bk', 'sd', 'sd', 'sd', 'b', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd',
                  'b', 'sv', 'sd', 'sd', 'b', 'sd', 'aa', 'aa', 'sd', 'sd', '%', 'sv', 'sd', 'b', 'sd', 'sv', 'sd',
                  'sv', 'b', 'qy', 'ny', 'sd', 'sd', 'x', 'b', 'sd', 'b', 'sv', 'b', 'sv', 'b', 'sd', 'sd', 'b', 'b',
                  'sv', 'b', 'aa', 'b', 'sv', 'sv', 'b', 'sv', 'qw', '%', 'sd', 'qy^d', 'b', '%', 'sd', '%', 'bf',
                  'aa', 'sd', 'sd', '%', 'sd', 'b', 'aa', 'qy', 'sd', 'br', 'sd', 'sd']
ground_truth = ['o', 'qw', '^h', 'sv', 'qo', 'sd', 'sd', 'sv', 'b', 'qy', 'nn', 'sd^e', 'sd', 'bk', 'qy', 'qo', '%',
                'sd', 'b', 'sd', 'sd', 'sd', 'b', 'sd', 'sd', 'sd', 'b', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd',
                'sd', 'b', 'sv', 'sd', 'sd', 'b', 'bf', 'ny', 'ny^r', 'sd^e', 'sd', '%', 'sv', 'sd', 'b', 'sv', 'sv',
                'sd', 'sd', 'b', 'bh', 'ny', 'sd^e', 'sd', 'x', 'b', 'ba', 'b', 'sv', 'b', 'sv', 'b', 'sd', 'sd', 'b',
                'b', 'sd', 'b', 'b', 'b^r', 'sd', 'sv', 'aa', 'sv', 'qo', '%', 'sd', 'qy^d', 'na', '%', 'sd', '%',
                'b^m', 'b', 'sd', 'sd', '%', 'h', 'aa', 'aa', 'qy', 'sd', 'b', 'sd', 'sd']

compare = lambda x, y: Counter(x) == Counter(y)
print(compare(non_con_pred, ground_truth))
print(compare(con_prediction, ground_truth))

for i in range(len(ground_truth)):
    print(Xtest[i])
    print('ground_truth - non_con_pred - con_prediction')
    print(ground_truth[i], '   -   ', non_con_pred[i], '   -   ', con_prediction[i])
