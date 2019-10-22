import os
from functools import reduce


def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

path_tree = get_directory_structure("IEMOCAP")

sessions =['S1', 'S2', 'S3', 'S4', 'S5']

for session in sessions:
    for text in path_tree[session]["transcriptions"]:
        file = open(text)

print('debug')