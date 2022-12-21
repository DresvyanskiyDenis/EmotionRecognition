import os
from functools import partial
from typing import Tuple, List

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import DeiTImageProcessor


from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader
from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image

import training_config
build_in_preprocessing_function = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")



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

def transform_emo_categories_to_int(df:pd.DataFrame, emo_categories:dict)->dict:
    pass



def load_all_dataframes():
    # TODO: check it
    path_to_AFEW_VA = r"G:\Datasets\AFEW-VA\AFEW-VA\AFEW-VA\preprocessed".replace("\\", os.sep)
    path_to_AffectNet = r"G:\Datasets\AffectNet\AffectNet\preprocessed".replace("\\", os.sep)
    path_to_RECOLA = r"G:\Datasets\RECOLA\preprocessed".replace("\\", os.sep)
    path_to_SEMAINE = r"G:\Datasets\SEMAINE\preprocessed".replace("\\", os.sep)
    path_to_SEWA = r"G:\Datasets\SEWA\preprocessed".replace("\\", os.sep)

    # load dataframes and add np.NaN labels to columns that are not present in the dataset
    AFEW_VA = pd.read_csv(os.path.join(path_to_AFEW_VA,"labels.csv"))
    RECOLA = pd.read_csv(os.path.join(path_to_RECOLA + "labels.csv"))
    SEMAINE = pd.read_csv(os.path.join(path_to_SEMAINE + "labels.csv"))
    SEWA = pd.read_csv(os.path.join(path_to_SEWA + "labels.csv"))

    AFEW_VA["category"] = np.NaN
    RECOLA["category"] = np.NaN
    SEMAINE["category"] = np.NaN
    SEWA["category"] = np.NaN


    # splitting to train, development, and test sets
    percentages = (80, 10, 10)
    AFEW_VA_train, AFEW_VA_dev, AFEW_VA_test = split_dataset_into_train_dev_test(AFEW_VA, percentages)
    RECOLA_train, RECOLA_dev, RECOLA_test = split_dataset_into_train_dev_test(RECOLA, percentages)
    SEMAINE_train, SEMAINE_dev, SEMAINE_test = split_dataset_into_train_dev_test(SEMAINE, percentages)
    SEWA_train, SEWA_dev, SEWA_test = split_dataset_into_train_dev_test(SEWA, percentages)
    # for the AffectNet we need to do a splitting separately, since it has no video, just a lot of images
    AffectNet_train = pd.read_csv(os.path.join(path_to_AffectNet, "train_labels.csv"))
    AffectNet_dev = pd.read_csv(os.path.join(path_to_AffectNet, "dev_labels.csv"))

    # concatenate all dataframes
    train = pd.concat([AFEW_VA_train, RECOLA_train, SEMAINE_train, SEWA_train, AffectNet_train])
    dev = pd.concat([AFEW_VA_dev, RECOLA_dev, SEMAINE_dev, SEWA_dev, AffectNet_dev])
    test = pd.concat([AFEW_VA_test, RECOLA_test, SEMAINE_test, SEWA_test])

    return (train, dev, test)


def preprocess_image_DeiT(image:np.ndarray)->torch.Tensor:
    image = build_in_preprocessing_function(images=image, return_tensors="pt")['pixel_values']
    return image

def resize_image_to_224(image:np.ndarray)->np.ndarray:
    # TODO: implement it in a way you wanted: black colour to make the image not distorted, etc.
    pass

def construct_data_loaders(train, dev, test, num_workers:int=8)\
        ->Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ # TODO: write a docstring

    :param train:
    :param dev:
    :param test:
    :param augment_prob:
    :param batch_size:
    :param num_workers:
    :return:
    """
    augmentation_functions = {
        pad_image_random_factor: training_config.AUGMENT_PROB,
        grayscale_image: training_config.AUGMENT_PROB,
        partial(collor_jitter_image_random, brightness=0.5, hue=0.3, contrast=0.3, saturation=0.3): training_config.AUGMENT_PROB,
        partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): training_config.AUGMENT_PROB,
        random_perspective_image: training_config.AUGMENT_PROB,
        random_rotation_image: training_config.AUGMENT_PROB,
        partial(random_crop_image, cropping_factor_limits=(0.7, 0.9)): training_config.AUGMENT_PROB,
        random_posterize_image: training_config.AUGMENT_PROB,
        partial(random_adjust_sharpness_image, sharpness_factor_limits=(0.1, 3)): training_config.AUGMENT_PROB,
        random_equalize_image: training_config.AUGMENT_PROB,
        random_horizontal_flip_image: training_config.AUGMENT_PROB,
        random_vertical_flip_image: training_config.AUGMENT_PROB
    }

    preprocessing_functions=[
        resize_image_to_224,
        preprocess_image_DeiT
    ]

    train_data_loader = ImageDataLoader(paths_with_labels=train, preprocessing_functions=preprocessing_functions,
                 augmentation_functions=augmentation_functions, shuffle=True)

    dev_data_loader = ImageDataLoader(paths_with_labels=dev, preprocessing_functions=preprocessing_functions,
                    augmentation_functions=None, shuffle=False)

    test_data_loader = ImageDataLoader(paths_with_labels=test, preprocessing_functions=preprocessing_functions,
                    augmentation_functions=None, shuffle=False)

    train_dataloader = DataLoader(train_data_loader, batch_size=training_config.BATCH_SIZE, num_workers=num_workers)
    dev_dataloader = DataLoader(dev_data_loader, batch_size=training_config.BATCH_SIZE, num_workers=num_workers)
    test_dataloader = DataLoader(test_data_loader, batch_size=training_config.BATCH_SIZE, num_workers=num_workers)

    return (train_dataloader, dev_dataloader, test_dataloader)

if __name__ == "__main__":
    load_all_dataframes()
