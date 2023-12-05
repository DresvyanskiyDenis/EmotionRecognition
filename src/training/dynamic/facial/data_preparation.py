import os
from functools import partial
from typing import Tuple, List, Dict, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader
from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor

splitting_seed: int = 101095


def load_all_dataframes(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    AFEW_VA = pd.read_csv(os.path.join(path_to_AFEW_VA, "labels.csv"))
    RECOLA = pd.read_csv(os.path.join(path_to_RECOLA, "preprocessed_labels.csv"))
    SEMAINE = pd.read_csv(os.path.join(path_to_SEMAINE, "preprocessed_labels.csv"))
    SEWA = pd.read_csv(os.path.join(path_to_SEWA, "preprocessed_labels.csv"))

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

    # unfortunately, we do not know the FPS of the AFEW-VA dataset. Other three dataset have 5 FPS as we have
    # downsampled them before. We assume that AFEW-VA has 25 FPS as it is the most common FPS in the wild.
    # therefore, we need to downsample AFEW-VA to 5 FPS. This means that we take every 5th frame from the dataset (approx.)
    AFEW_VA = AFEW_VA.iloc[::5, :]

    # splitting to train, development, and test sets
    percentages = (80, 10, 10)
    AFEW_VA_train, AFEW_VA_dev, AFEW_VA_test = split_dataset_into_train_dev_test(AFEW_VA, percentages,
                                                                                 seed=splitting_seed)
    RECOLA_train, RECOLA_dev, RECOLA_test = split_dataset_into_train_dev_test(RECOLA, percentages, seed=splitting_seed)
    SEMAINE_train, SEMAINE_dev, SEMAINE_test = split_dataset_into_train_dev_test(SEMAINE, percentages,
                                                                                 seed=splitting_seed)
    SEWA_train, SEWA_dev, SEWA_test = split_dataset_into_train_dev_test(SEWA, percentages, seed=splitting_seed)

    # concatenate all dataframes
    train = pd.concat([AFEW_VA_train, RECOLA_train, SEMAINE_train, SEWA_train])
    dev = pd.concat([AFEW_VA_dev, RECOLA_dev, SEMAINE_dev, SEWA_dev])
    test = pd.concat([AFEW_VA_test, RECOLA_test, SEMAINE_test, SEWA_test])
    # change external_hdd_1 to external_hdd_2 in paths for all datasets
    train['path'] = train['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    dev['path'] = dev['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    test['path'] = test['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    # again, change paths from "/media/external_hdd_2/Datasets" to  "/work/home/dsu/Datasets"
    train['path'] = train['path'].apply(
        lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    dev['path'] = dev['path'].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    test['path'] = test['path'].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))

    return (train, dev, test)


def load_all_datasets_embeddings() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path_to_AFEW_VA_train_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/AFEW-VA/afew_va_train.csv"
    path_to_AFEW_VA_dev_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/AFEW-VA/afew_va_dev.csv"
    path_to_AFEW_VA_test_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/AFEW-VA/afew_va_test.csv"
    path_to_RECOLA_train_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/RECOLA/recola_train.csv"
    path_to_RECOLA_dev_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/RECOLA/recola_dev.csv"
    path_to_RECOLA_test_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/RECOLA/recola_test.csv"
    path_to_SEMAINE_train_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/SEMAINE/semaine_train.csv"
    path_to_SEMAINE_dev_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/SEMAINE/semaine_dev.csv"
    path_to_SEMAINE_test_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/SEMAINE/semaine_test.csv"
    path_to_SEWA_train_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/SEWA/sewa_train.csv"
    path_to_SEWA_dev_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/SEWA/sewa_dev.csv"
    path_to_SEWA_test_embeddings = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/SEWA/sewa_test.csv"

    # load all datasets
    AFEW_VA_train = pd.read_csv(path_to_AFEW_VA_train_embeddings)
    AFEW_VA_dev = pd.read_csv(path_to_AFEW_VA_dev_embeddings)
    AFEW_VA_test = pd.read_csv(path_to_AFEW_VA_test_embeddings)
    RECOLA_train = pd.read_csv(path_to_RECOLA_train_embeddings)
    RECOLA_dev = pd.read_csv(path_to_RECOLA_dev_embeddings)
    RECOLA_test = pd.read_csv(path_to_RECOLA_test_embeddings)
    SEMAINE_train = pd.read_csv(path_to_SEMAINE_train_embeddings)
    SEMAINE_dev = pd.read_csv(path_to_SEMAINE_dev_embeddings)
    SEMAINE_test = pd.read_csv(path_to_SEMAINE_test_embeddings)
    SEWA_train = pd.read_csv(path_to_SEWA_train_embeddings)
    SEWA_dev = pd.read_csv(path_to_SEWA_dev_embeddings)
    SEWA_test = pd.read_csv(path_to_SEWA_test_embeddings)

    # concat them in train, dev, test
    train = pd.concat([AFEW_VA_train, RECOLA_train, SEMAINE_train, SEWA_train])
    dev = pd.concat([AFEW_VA_dev, RECOLA_dev, SEMAINE_dev, SEWA_dev])
    test = pd.concat([AFEW_VA_test, RECOLA_test, SEMAINE_test, SEWA_test])

    # load train, dev, test with old function to get labels. Ultra inefficient, but I need to do it fast and only
    # for one experiment
    old_train, old_dev, old_test = load_all_dataframes(seed=splitting_seed)
    # sort new and old train, dev, test by path so that then we can copy labels
    train = train.sort_values(by=['path']).reset_index(drop=True)
    dev = dev.sort_values(by=['path']).reset_index(drop=True)
    test = test.sort_values(by=['path']).reset_index(drop=True)
    old_train = old_train.sort_values(by=['path']).reset_index(drop=True)
    old_dev = old_dev.sort_values(by=['path']).reset_index(drop=True)
    old_test = old_test.sort_values(by=['path']).reset_index(drop=True)
    # copy labels
    train["arousal"] = old_train["arousal"]
    train["valence"] = old_train["valence"]
    dev["arousal"] = old_dev["arousal"]
    dev["valence"] = old_dev["valence"]
    test["arousal"] = old_test["arousal"]
    test["valence"] = old_test["valence"]
    return (train, dev, test)


def split_dataset_into_train_dev_test(filenames_labels: pd.DataFrame, percentages: Tuple[int, int, int],
                                      seed: int = 42) -> List[pd.DataFrame]:
    # get unique filenames (videos)
    filenames_labels = filenames_labels.copy().dropna(subset=['path'])
    filenames_labels['path'] = filenames_labels['path'].astype(str)
    if "AFEW-VA" in filenames_labels['path'].iloc[0]:
        filenames = filenames_labels['path'].apply(lambda x: '/' + x.split(os.path.sep)[-2] + '/')
    elif "SEWA" in filenames_labels['path'].iloc[0] or "RECOLA" in filenames_labels['path'].iloc[0] \
            or "SEMAINE" in filenames_labels['path'].iloc[0]:
        filenames = filenames_labels['path'].apply(
            lambda x: x.split(os.path.sep)[-1].split("_")[0] + "_")  # pattern: videoName_
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
    dev_filenames = filenames[indicies[num_train:num_train + num_dev]]
    test_filenames = filenames[indicies[num_train + num_dev:]]

    # get dataframes for each set
    train = pd.concat([filenames_dict[filename] for filename in train_filenames])
    dev = pd.concat([filenames_dict[filename] for filename in dev_filenames])
    test = pd.concat([filenames_dict[filename] for filename in test_filenames])
    return [train, dev, test]


def get_augmentation_function(probability: float) -> Dict[Callable, float]:
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


def separate_various_videos(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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
    results['videoname'] = results['path'].apply(lambda x: x.split(os.path.sep)[-1].split("_")[0]
    if "SEWA" in x or "RECOLA" in x or "SEMAINE" in x
    else x.split(os.path.sep)[-2])
    results = results.groupby('videoname')
    results = {name: group for name, group in results}
    return results


def construct_data_loaders(train: Dict[str, pd.DataFrame], dev: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame],
                           window_size: Union[int, float], stride: Union[int, float],
                           preprocessing_functions: List[Callable],
                           batch_size: int,
                           augmentation_functions: Optional[Dict[Callable, float]] = None,
                           num_workers: int = 8) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Constructs torch.utils.data.DataLoader for train, dev, and test sets. The TemporalDataLoader is used for
        constructing the data loaders as it cuts the data on temporal windows internally.
    Args:
        train: pd.DataFrame
            The dataframe with train data.
        dev: pd.DataFrame
            The dataframe with development data.
        test: pd.DataFrame
            The dataframe with test data.
        preprocessing_functions: List[Callable]
            A list of preprocessing functions to apply to the images.
        batch_size: int
            The batch size.
        augmentation_functions: Optional[Dict[Callable, float]]
            A dictionary of augmentation functions and the probabilities of their application to every image.
            If None, no augmentation is applied.
        num_workers: int
            The number of workers to use for loading and preprocessinf the data.
    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        A tuple of train, dev, and test data loaders.
    """
    labels_columns = ['arousal', 'valence']
    # we need the TemporalEmbeddingsLoader class (from datatools) that uses already extracted embeddings
    # instead of loading images as TemporalDataLoader does
    train_data_loader = TemporalEmbeddingsLoader(embeddings_with_labels=train, label_columns=labels_columns,
                                                 feature_columns=['embedding_%i' % i for i in range(256)],
                                                 window_size=window_size, stride=stride,
                                                 consider_timestamps=False,
                                                 preprocessing_functions=preprocessing_functions, shuffle=True)

    dev_data_loader = TemporalEmbeddingsLoader(embeddings_with_labels=dev, label_columns=labels_columns,
                                               feature_columns=['embedding_%i' % i for i in range(256)],
                                               window_size=window_size, stride=stride,
                                               consider_timestamps=False,
                                               preprocessing_functions=preprocessing_functions, shuffle=False)

    test_data_loader = TemporalEmbeddingsLoader(embeddings_with_labels=test, label_columns=labels_columns,
                                                feature_columns=['embedding_%i' % i for i in range(256)],
                                                window_size=window_size, stride=stride,
                                                consider_timestamps=False,
                                                preprocessing_functions=preprocessing_functions, shuffle=False)

    train_dataloader = DataLoader(train_data_loader, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  shuffle=True)
    dev_dataloader = DataLoader(dev_data_loader, batch_size=batch_size, num_workers=num_workers // 2, shuffle=False)
    test_dataloader = DataLoader(test_data_loader, batch_size=batch_size, num_workers=num_workers // 2, shuffle=False)

    return (train_dataloader, dev_dataloader, test_dataloader)


def get_data_loaders(window_size: Union[float, int], stride: Union[float, int],
                     base_model_type: str, batch_size: int):
    """
    Loads data, constructs data loaders, and returns them.
    Args:
        base_model_type: str
            The type of the base model to get the preprocessing functions for it. Can be 'EfficientNet-B1',
            'EfficientNet-B4'
        batch_size: int
            The batch size.

    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        A tuple of train, dev, and test data loaders.
    """
    # load data and separate it on videos
    train, dev, test = load_all_datasets_embeddings()
    train, dev, test = separate_various_videos(train), separate_various_videos(dev), separate_various_videos(test)
    # get preprocessing functions
    if base_model_type == 'EfficientNet-B1':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif base_model_type == 'EfficientNet-B4':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    else:
        raise ValueError(f'The model type should be either "EfficientNet-B1", "EfficientNet-B4"'
                         f'Got {base_model_type} instead.')
    # As we load embeddings, we do not need any preprocessing functions or augmentation
    preprocessing_functions = None
    augmentation_functions = None
    # construct data loaders
    train_dataloader, dev_dataloader, test_dataloader = construct_data_loaders(train, dev, test,
                                                                               window_size=window_size,
                                                                               stride=stride,
                                                                               preprocessing_functions=preprocessing_functions,
                                                                               batch_size=batch_size,
                                                                               augmentation_functions=augmentation_functions)

    return (train_dataloader, dev_dataloader, test_dataloader)


if __name__ == "__main__":
    train, dev, test = get_data_loaders(window_size=10, stride=4, base_model_type='EfficientNet-B1', batch_size=32)
    print(train.__len__(), dev.__len__(), test.__len__())
    print('-----------------------')
    for x, y in train:
        print(x.shape, y.shape)
        break
