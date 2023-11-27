import os
from functools import partial
from typing import Tuple, List, Dict, Callable

import numpy as np
import pandas as pd

from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image


def load_all_dataframes(seed:int=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
     Loads all dataframes for the datasets AFEW-VA, AffectNet, RECOLA, SEMAINE, and SEWA, and split them into
        train, dev, and test sets.
    Returns: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The tuple of train, dev, and test data.

    """
    path_to_AFEW_VA = r"/work/home/dsu/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed".replace("\\", os.sep)
    path_to_RECOLA = r"/work/home/dsu/Datasets/RECOLA/preprocessed".replace("\\", os.sep)
    path_to_SEMAINE = r"/work/home/dsu/Datasets/SEMAINE/preprocessed".replace("\\", os.sep)
    path_to_SEWA = r"/work/home/dsu/Datasets/SEWA/preprocessed".replace("\\", os.sep)

    # load dataframes and add np.NaN labels to columns that are not present in the dataset
    AFEW_VA = pd.read_csv(os.path.join(path_to_AFEW_VA,"labels.csv"))
    RECOLA = pd.read_csv(os.path.join(path_to_RECOLA,"preprocessed_labels.csv"))
    SEMAINE = pd.read_csv(os.path.join(path_to_SEMAINE,"preprocessed_labels.csv"))
    SEWA = pd.read_csv(os.path.join(path_to_SEWA,"preprocessed_labels.csv"))

    # change column names of the datasets to "path" from "frame_num"
    AFEW_VA = AFEW_VA.rename(columns={"frame_num": "path"})
    RECOLA = RECOLA.rename(columns={"filename": "path"})
    SEMAINE = SEMAINE.rename(columns={"filename": "path"})
    SEWA = SEWA.rename(columns={"filename": "path"})
    # drop timestamp from RECOLA, SEMAINE, and SEWA
    RECOLA = RECOLA.drop(columns=['timestamp'])
    SEMAINE = SEMAINE.drop(columns=['timestamp'])
    SEWA = SEWA.drop(columns=['timestamp'])
    # transform valence and arousal values from [-10, 10] range to [-1, 1] range for AFEW-VA dataset
    AFEW_VA['valence'] = AFEW_VA['valence'].apply(lambda x: x / 10.)
    AFEW_VA['arousal'] = AFEW_VA['arousal'].apply(lambda x: x / 10.)

    # unfortunately, we do not know the FPS of the AFEW-VA dataset. Other three dataset have 3 FPS as we have
    # downsampled them before. We assume that AFEW-VA has 25 FPS as it is the most common FPS in the wild.
    # therefore, we need to downsample AFEW-VA to 3 FPS. This means that we take every 8th frame from the dataset (approx.)
    AFEW_VA = AFEW_VA.iloc[::8, :]

    # splitting to train, development, and test sets
    percentages = (80, 10, 10)
    AFEW_VA_train, AFEW_VA_dev, AFEW_VA_test = split_dataset_into_train_dev_test(AFEW_VA, percentages, seed=seed)
    RECOLA_train, RECOLA_dev, RECOLA_test = split_dataset_into_train_dev_test(RECOLA, percentages, seed=seed)
    SEMAINE_train, SEMAINE_dev, SEMAINE_test = split_dataset_into_train_dev_test(SEMAINE, percentages, seed=seed)
    SEWA_train, SEWA_dev, SEWA_test = split_dataset_into_train_dev_test(SEWA, percentages, seed=seed)

    # concatenate all dataframes
    train = pd.concat([AFEW_VA_train, RECOLA_train, SEMAINE_train, SEWA_train])
    dev = pd.concat([AFEW_VA_dev, RECOLA_dev, SEMAINE_dev, SEWA_dev])
    test = pd.concat([AFEW_VA_test, RECOLA_test, SEMAINE_test, SEWA_test])
    # change external_hdd_1 to external_hdd_2 in paths for all datasets
    train['path'] = train['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    dev['path'] = dev['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    test['path'] = test['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    # again, change paths from "/media/external_hdd_2/Datasets" to  "/work/home/dsu/Datasets"
    train['path'] = train['path'].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    dev['path'] = dev['path'].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    test['path'] = test['path'].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))

    return (train, dev, test)


def split_dataset_into_train_dev_test(filenames_labels:pd.DataFrame, percentages:Tuple[int,int,int],
                                      seed:int=42)->List[pd.DataFrame]:
    # get unique filenames (videos)
    filenames_labels = filenames_labels.copy().dropna(subset=['path'])
    filenames_labels['path'] = filenames_labels['path'].astype(str)
    if "AFEW-VA" in filenames_labels['path'].iloc[0]:
        filenames =filenames_labels['path'].apply(lambda x: '/'+x.split(os.path.sep)[-2]+'/')
    elif "SEWA" in filenames_labels['path'].iloc[0] or "RECOLA" in filenames_labels['path'].iloc[0]\
        or "SEMAINE" in filenames_labels['path'].iloc[0]:
        filenames = filenames_labels['path'].apply(lambda x: x.split(os.path.sep)[-1].split("_")[0]+"_") # pattern: videoName_
    else:
        raise ValueError("This function only supports AFEW-VA, SEWA, RECOLA, SAVEE, and SEMAINE datasets.")
    filenames = pd.unique(filenames)
    # divide dataset into unique filenames as a dictionary
    filenames_dict = {}
    for filename in filenames:

        filenames_dict[filename] = filenames_labels[filenames_labels['path'].str.contains(filename)]
    # divide dataset into train, development, and test sets
    num_train = int(round(len(filenames_dict) * percentages[0] / 100))
    num_dev = int(round(len(filenames_dict) * percentages[1] / 100))
    num_test = len(filenames_dict) - num_train - num_dev
    if num_train + num_dev + num_test != len(filenames_dict):
        raise ValueError("One or more entities in the filename_dict have been lost during splitting.")
    # get random filenames for each set
    indicies = np.random.RandomState(seed=seed).permutation(len(filenames_dict))
    train_filenames = filenames[indicies[:num_train]]
    dev_filenames = filenames[indicies[num_train:num_train+num_dev]]
    test_filenames = filenames[indicies[num_train+num_dev:]]

    # get dataframes for each set
    train = pd.concat([filenames_dict[filename] for filename in train_filenames])
    dev = pd.concat([filenames_dict[filename] for filename in dev_filenames])
    test = pd.concat([filenames_dict[filename] for filename in test_filenames])
    return [train, dev, test]


def get_augmentation_function(probability:float)->Dict[Callable, float]:
    """
    Returns a dictionary of augmentation functions and the probabilities of their application.
    Args:
        probability: float
            The probability of applying the augmentation function.

    Returns: Dict[Callable, float]
        A dictionary of augmentation functions and the probabilities of their application.

    """
    augmentation_functions = {
        pad_image_random_factor: probability,
        grayscale_image: probability,
        partial(collor_jitter_image_random, brightness=0.5, hue=0.3, contrast=0.3,
                saturation=0.3): probability,
        partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): probability,
        random_perspective_image: probability,
        random_rotation_image: probability,
        partial(random_crop_image, cropping_factor_limits=(0.7, 0.9)): probability,
        random_posterize_image: probability,
        partial(random_adjust_sharpness_image, sharpness_factor_limits=(0.1, 3)): probability,
        random_equalize_image: probability,
        random_horizontal_flip_image: probability,
        random_vertical_flip_image: probability,
    }
    return augmentation_functions

def separate_various_videos(data:pd.DataFrame)->Dict[str, pd.DataFrame]:
    """
    Separates the provided dataframe onto different videos represented as dataframes (that contain frames).
    :param data: pd.DataFrame
        The dataframe to separate.
    :return: Dict[str, pd.DataFrame]
        A dictionary of video names and their corresponding dataframes.
    """
    # SEWA, RECOLA, SEMAINE have the format of the path: ***/videoname_sec_msec.png
    # AFEW-VA has the format of the path: ***/videoname/frame_num.png
    # therefore, we will separate the dataframes by videonames, but before that we need to
    # extract the videonames from the paths
    results = data.copy(deep=True)
    results['videoname']=results['path'].apply(lambda x: x.split(os.path.sep)[-1].split("_")[0]
                                                    if "SEWA" in x or "RECOLA" in x or "SEMAINE" in x
                                                    else x.split(os.path.sep)[-2])
    results = results.groupby('videoname')
    results = {name:group for name, group in results}
    return results



if __name__=="__main__":
    train, dev, test = load_all_dataframes()
    res = separate_various_videos(train)
    a=1+2.
    print(a)
