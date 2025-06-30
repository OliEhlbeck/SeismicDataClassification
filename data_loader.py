import json
import random

from sklearn.model_selection import train_test_split

def load_data(use_z_only : bool):
    if use_z_only == True:
        file_path = "D:/Oli/Uni/23_24 WiSe/MachLearningGeoPhy/datasetPycharm/datasetPycharm_z.json"
    else:
        file_path = "D:/Oli/Uni/23_24 WiSe/MachLearningGeoPhy/datasetPycharm/datasetPycharm_all.json"

    print("Load and prepare dataset ... ")

    dataset = json.load(open(file_path, 'r'))

    random.shuffle(dataset)
    dataset = dataset[:50000]

    # Split into X and Y. X contains features, Y contains class-label
    X = [e[:-1] for e in dataset]
    Y = [e[-1] for e in dataset]

    # Split into training and testing
    print("Split into training and testing ... ")
    return train_test_split(X, Y, test_size=0.2, random_state=42)

