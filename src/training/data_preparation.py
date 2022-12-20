import os
from typing import Tuple, List

import pandas as pd
import numpy as np


def split_dataset_into_train_dev_test(filenames_labels:pd.DataFrame, percentages:Tuple[int,int,int])->List[pd.DataFrame]:
    # get unique filenames (videos)
    filenames_labels = filenames_labels.dropna()
    filenames_labels['path'] = filenames_labels['path'].apply(lambda x: x.replace("\\", os.path.sep))
    filenames_labels['path'] = filenames_labels['path'].astype("str")
    filenames = filenames_labels['path'].apply(lambda x: x.split(os.path.sep)[-1].split("_")[0]+"_") # pattern: videoName_
    filenames = pd.unique(filenames)
    # divide dataset into unique filenames as a dictionary
    filenames_dict = {}
    for filename in filenames:
        # TODO: check it
        filenames_dict[filename] = filenames_labels[filenames_labels['path'].str.contains(filename)]
    # divide dataset into train, development, and test sets
    num_train = int(round(len(filenames_dict) * percentages[0] / 100))
    num_dev = int(round(len(filenames_dict) * percentages[1] / 100))
    num_test = len(filenames_dict) - num_train - num_dev
    if num_train + num_dev + num_test != len(filenames_dict):
        raise ValueError("One or more entities in the filename_dict have been lost during splitting.")
    # get random filenames for each set
    train_filenames = np.random.choice(filenames, num_train, replace=False)
    dev_filenames = np.random.choice(filenames, num_dev, replace=False)
    test_filenames = np.random.choice(filenames, num_test, replace=False)
    # get dataframes for each set
    train = pd.concat([filenames_dict[filename] for filename in train_filenames])
    dev = pd.concat([filenames_dict[filename] for filename in dev_filenames])
    test = pd.concat([filenames_dict[filename] for filename in test_filenames])
    return [train, dev, test]





def load_all_dataframes():
    path_to_AFEW_VA = r"G:\Datasets\AFEW-VA\AFEW-VA\AFEW-VA\preprocessed".replace("\\", os.sep)
    path_to_AffectNet = r"G:\Datasets\AffectNet\AffectNet\preprocessed".replace("\\", os.sep)
    path_to_RECOLA = r"G:\Datasets\RECOLA\preprocessed".replace("\\", os.sep)
    path_to_SEMAINE = r"G:\Datasets\SEMAINE\preprocessed".replace("\\", os.sep)
    path_to_SEWA = r"G:\Datasets\SEWA\preprocessed".replace("\\", os.sep)

    AFEW_VA = pd.read_csv(os.path.join(path_to_AFEW_VA,"labels.csv"))
    #AffectNet = pd.read_csv(path_to_AffectNet + "labels.csv")
    #RECOLA = pd.read_csv(path_to_RECOLA + "labels.csv")
    #SEMAINE = pd.read_csv(path_to_SEMAINE + "labels.csv")
    #SEWA = pd.read_csv(path_to_SEWA + "labels.csv")

    # splitting to train, development, and test sets
    percentages = (80, 10, 10)
    AFEW_VA_train, AFEW_VA_dev, AFEW_VA_test = split_dataset_into_train_dev_test(AFEW_VA, percentages)


def construct_data_loaders():
    pass



if __name__ == "__main__":
    load_all_dataframes()
