from src.mocap_utils import *
import csv
import pandas as pd


def get_mrda_data(write=False, read_from_csv=False, csv_file_name="data/MRDA/mrda_dataset.csv"):

    """
    get_directory_structure("IEMOCAP") - Sorts differently on Windows and Ubuntu,
    hence better to save csv after we read all data, and use that one latter
    :param write: write the csv file
    :param csv_file_name: provide correct path
    :param read_from_csv: read from the written file
    :return: lists of utterances, emo_dialogues, emo_evo, v, a, d, utt_keys
    """

    utterances = []
    da = []
    utt_keys = []
    fieldnames = ['utt_keys', 'utterance', 'da']

    if not read_from_csv:
        if write:
            store_mocap_in_csv = open(csv_file_name, mode='w', newline='')
            writer = csv.DictWriter(store_mocap_in_csv, fieldnames=fieldnames)
            writer.writeheader()

        path = "data/MRDA/"
        path_tree = get_directory_structure(path)

        # data_split
        train_set_idx = ['Bdb001', 'Bed002', 'Bed004', 'Bed005', 'Bed008', 'Bed009', 'Bed011', 'Bed013', 'Bed014',
                         'Bed015', 'Bed017', 'Bmr002', 'Bmr003', 'Bmr006', 'Bmr007', 'Bmr008', 'Bmr009', 'Bmr011',
                         'Bmr012', 'Bmr015', 'Bmr016', 'Bmr020', 'Bmr021', 'Bmr023', 'Bmr025', 'Bmr026', 'Bmr027',
                         'Bmr029', 'Bmr031', 'Bns001', 'Bns002', 'Bns003', 'Bro003', 'Bro005', 'Bro007', 'Bro010',
                         'Bro012', 'Bro013', 'Bro015', 'Bro016', 'Bro017', 'Bro019', 'Bro022', 'Bro023', 'Bro025',
                         'Bro026', 'Bro028', 'Bsr001', 'Btr001', 'Btr002', 'Buw001']
        valid_set_idx = ['Bed003', 'Bed010', 'Bmr005', 'Bmr014', 'Bmr019', 'Bmr024', 'Bmr030', 'Bro004', 'Bro011',
                         'Bro018', 'Bro024']
        test_set_idx = ['Bed006', 'Bed012', 'Bed016', 'Bmr001', 'Bmr010', 'Bmr022', 'Bmr028', 'Bro008', 'Bro014',
                        'Bro021', 'Bro027']

        for set in test_set_idx:
            filename = path + set + ".out"
            f = open(filename, 'r').read()
            f = np.array(f.split('\n'))
            for line in f:
                line.split(",")

    return utterances, da, utt_keys


utterances, da, utt_keys = get_mrda_data()
print('debug')
