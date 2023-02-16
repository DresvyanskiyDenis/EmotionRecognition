import os
from functools import partial
from typing import Tuple, List, Callable, Optional, Dict, Union

import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch.utils.data import DataLoader
from torchvision.io import read_image
from transformers import DeiTImageProcessor

from decorators.common_decorators import timer
from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader
from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image

import training_config
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor, \
    resize_image_to_224_saving_aspect_ratio, preprocess_image_MobileNetV3

DeiT_preprocessing_function = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")



def split_dataset_into_train_dev_test(filenames_labels:pd.DataFrame, percentages:Tuple[int,int,int],
                                      seed:int=42)->List[pd.DataFrame]:
    # get unique filenames (videos)
    filenames_labels = filenames_labels.copy().dropna()
    filenames_labels['path'] = filenames_labels['path'].astype(str)
    if "AFEW-VA" in filenames_labels['path'].iloc[0]:
        filenames =filenames_labels['path'].apply(lambda x: '/'+x.split(os.path.sep)[-2]+'/')
    elif "SEWA" in filenames_labels['path'].iloc[0] or "RECOLA" in filenames_labels['path'].iloc[0]\
        or "SEMAINE" in filenames_labels['path'].iloc[0]:
        filenames = filenames_labels['path'].apply(lambda x: x.split(os.path.sep)[-1].split("_")[0]+"_") # pattern: videoName_
    else:
        raise ValueError("This function only supports AFEW-VA, SEWA, RECOLA, and SEMAINE datasets.")
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

def transform_emo_categories_to_int(df:pd.DataFrame, emo_categories:dict)->dict:
    df['category'] = df['category'].apply(lambda x: emo_categories[x])
    return df



def load_all_dataframes(seed:int=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
     Loads all dataframes for the datasets AFEW-VA, AffectNet, RECOLA, SEMAINE, and SEWA, and split them into
        train, dev, and test sets.
    Returns: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The tuple of train, dev, and test data.

    """
    path_to_AFEW_VA = r"/work/home/dsu/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed".replace("\\", os.sep)
    path_to_AffectNet = r"/work/home/dsu/Datasets/AffectNet/AffectNet/preprocessed".replace("\\", os.sep)
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



    # splitting to train, development, and test sets
    percentages = (80, 10, 10)
    AFEW_VA_train, AFEW_VA_dev, AFEW_VA_test = split_dataset_into_train_dev_test(AFEW_VA, percentages, seed=seed)
    RECOLA_train, RECOLA_dev, RECOLA_test = split_dataset_into_train_dev_test(RECOLA, percentages, seed=seed)
    SEMAINE_train, SEMAINE_dev, SEMAINE_test = split_dataset_into_train_dev_test(SEMAINE, percentages, seed=seed)
    SEWA_train, SEWA_dev, SEWA_test = split_dataset_into_train_dev_test(SEWA, percentages, seed=seed)
    # add NaN values to 'category' column for the datasets that do not have it
    AFEW_VA_train['category'], AFEW_VA_dev['category'], AFEW_VA_test['category'] = np.NaN, np.NaN, np.NaN
    RECOLA_train['category'], RECOLA_dev['category'], RECOLA_test['category'] = np.NaN, np.NaN, np.NaN
    SEMAINE_train['category'], SEMAINE_dev['category'], SEMAINE_test['category'] = np.NaN, np.NaN, np.NaN
    SEWA_train['category'], SEWA_dev['category'], SEWA_test['category'] = np.NaN, np.NaN, np.NaN
    # for the AffectNet we need to do a splitting separately, since it has no video, just a lot of images
    AffectNet_train = pd.read_csv(os.path.join(path_to_AffectNet, "train_labels.csv"))
    AffectNet_train = transform_emo_categories_to_int(AffectNet_train, training_config.EMO_CATEGORIES)
    AffectNet_dev = pd.read_csv(os.path.join(path_to_AffectNet, "dev_labels.csv"))
    AffectNet_dev = transform_emo_categories_to_int(AffectNet_dev, training_config.EMO_CATEGORIES)
    # change columns name of AffectNet from abs_path to path
    AffectNet_train = AffectNet_train.rename(columns={"abs_path": "path"})
    AffectNet_dev = AffectNet_dev.rename(columns={"abs_path": "path"})
    # drop all images from AffectNet, which are mot jpg or png
    allowed_extensions = ['jpg', 'png', 'JPG', 'jpeg', 'PNG', 'Jpeg','JPEG']
    AffectNet_train = AffectNet_train[AffectNet_train['path'].apply(lambda x: x.split('.')[-1] in allowed_extensions)]
    AffectNet_dev = AffectNet_dev[AffectNet_dev['path'].apply(lambda x: x.split('.')[-1] in allowed_extensions)]



    # concatenate all dataframes
    train = pd.concat([AFEW_VA_train, RECOLA_train, SEMAINE_train, SEWA_train, AffectNet_train])
    dev = pd.concat([AFEW_VA_dev, RECOLA_dev, SEMAINE_dev, SEWA_dev, AffectNet_dev])
    test = pd.concat([AFEW_VA_test, RECOLA_test, SEMAINE_test, SEWA_test])
    # change external_hdd_1 to external_hdd_2 in paths for all datasets
    train['path'] = train['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    dev['path'] = dev['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))
    test['path'] = test['path'].apply(lambda x: x.replace("external_hdd_1", "external_hdd_2"))

    return (train, dev, test)


def preprocess_image_DeiT(image:np.ndarray)->torch.Tensor:
    image = DeiT_preprocessing_function(images=image, return_tensors="pt")['pixel_values']
    return image

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
        partial(gaussian_blur_image_random, kernel_size=(5, 9), sigma=(0.1, 5)): training_config.AUGMENT_PROB,
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


def construct_data_loaders(train:pd.DataFrame, dev:pd.DataFrame, test:pd.DataFrame,
                           preprocessing_functions:List[Callable],
                           augmentation_functions:Optional[Dict[Callable, float]]=None,
                           num_workers:int=8)\
        ->Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Constructs the data loaders for the train, dev and test sets.

    Args:
        train: pd.DataFrame
            The train set. It should contain the columns 'path' TODO: write columns.
        dev: pd.DataFrame
            The dev set. It should contain the columns 'path' TODO: write columns.
        test: pd.DataFrame
            The test set. It should contain the columns 'path' TODO: write columns.
        preprocessing_functions: List[Callable]
            A list of preprocessing functions to be applied to the images.
        augmentation_functions: Optional[Dict[Callable, float]]
            A dictionary of augmentation functions and the probabilities of their application.
        num_workers: int
            The number of workers to be used by the data loaders.

    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The data loaders for the train, dev and test sets.

    """

    train_data_loader = ImageDataLoader(paths_with_labels=train, preprocessing_functions=preprocessing_functions,
                 augmentation_functions=augmentation_functions, shuffle=True)

    dev_data_loader = ImageDataLoader(paths_with_labels=dev, preprocessing_functions=preprocessing_functions,
                    augmentation_functions=None, shuffle=False)

    test_data_loader = ImageDataLoader(paths_with_labels=test, preprocessing_functions=preprocessing_functions,
                    augmentation_functions=None, shuffle=False)

    train_dataloader = DataLoader(train_data_loader, batch_size=training_config.BATCH_SIZE, num_workers=num_workers, shuffle=True)
    dev_dataloader = DataLoader(dev_data_loader, batch_size=training_config.BATCH_SIZE, num_workers=num_workers//2, shuffle=False)
    test_dataloader = DataLoader(test_data_loader, batch_size=training_config.BATCH_SIZE, num_workers=num_workers//4, shuffle=False)

    return (train_dataloader, dev_dataloader, test_dataloader)


def load_data_and_construct_dataloaders(model_type:str, return_class_weights:Optional[bool]=False)->\
        Union[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader],
              Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]]:
    """
        Args:
            model_type: str
            The type of the model. It can be 'MobileNetV3_large' or 'EfficientNet-B1'.
            return_class_weights: Optional[bool]
            If True, the function returns the class weights as well.

    Loads the data presented in pd.DataFrames and constructs the data loaders using them. It is a general function
    to assemble all functions defined above.
    Returns: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The train, dev and test data loaders.
        or
        Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader], torch.Tensor]
        The train, dev and test data loaders and the class weights calculated based on the training labels.

    """
    if model_type not in ['MobileNetV3_large', 'EfficientNet-B1']:
        raise ValueError('The model type should be either "MobileNetV3_large" or "EfficientNet-B1".')
    # load pd.DataFrames
    train, dev, test = load_all_dataframes(training_config.splitting_seed)
    # define preprocessing functions
    if model_type == 'MobileNetV3_large':
        preprocessing_functions = [resize_image_to_224_saving_aspect_ratio,
                                   preprocess_image_MobileNetV3]
    elif model_type == 'EfficientNet-B1':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size = 240),
                                   EfficientNet_image_preprocessor()]
    else:
        raise ValueError(f'The model type should be either "MobileNetV3_large" or "EfficientNet-B1". Got {model_type} instead.')
    # define augmentation functions
    augmentation_functions = get_augmentation_function(training_config.AUGMENT_PROB)
    # construct data loaders
    train_dataloader, dev_dataloader, test_dataloader = construct_data_loaders(train, dev, test,
                                                                               preprocessing_functions,
                                                                               augmentation_functions,
                                                                               num_workers=training_config.NUM_WORKERS)

    if return_class_weights:
        num_classes = train.iloc[:, -1].nunique()
        labels = pd.DataFrame(train.iloc[:,-1])
        labels = labels.dropna()
        labels = labels.astype(int)
        class_weights = torch.nn.functional.one_hot(torch.tensor(labels.values), num_classes=num_classes)
        class_weights = class_weights.sum(axis=0)
        class_weights = 1. / (class_weights / class_weights.sum())
        # normalize class weights
        class_weights = class_weights / class_weights.sum()
        return ((train_dataloader, dev_dataloader, test_dataloader), class_weights)

    return (train_dataloader, dev_dataloader, test_dataloader)



@timer
def main():
    train_data_loader, dev_data_loader, test_data_loader = load_data_and_construct_dataloaders()
    for x, y in train_data_loader:
        print(x.shape, y.shape)
        print("-------------------")

if __name__ == "__main__":
    main()


    """path_to_AFEW_VA = r"/work/home/dsu/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed".replace("\\", os.sep)
    path_to_AffectNet = r"/work/home/dsu/Datasets/AffectNet/AffectNet/preprocessed".replace("\\", os.sep)
    path_to_RECOLA = r"/work/home/dsu/Datasets/RECOLA/preprocessed".replace("\\", os.sep)
    path_to_SEMAINE = r"/work/home/dsu/Datasets/SEMAINE/preprocessed".replace("\\", os.sep)
    path_to_SEWA = r"/work/home/dsu/Datasets/SEWA/preprocessed".replace("\\", os.sep)

    # load dataframes and add np.NaN labels to columns that are not present in the dataset
    AFEW_VA = pd.read_csv(os.path.join(path_to_AFEW_VA, "labels.csv"))
    RECOLA = pd.read_csv(os.path.join(path_to_RECOLA, "preprocessed_labels.csv"))
    SEMAINE = pd.read_csv(os.path.join(path_to_SEMAINE, "preprocessed_labels.csv"))
    SEWA = pd.read_csv(os.path.join(path_to_SEWA, "preprocessed_labels.csv"))
    AffectNet_train = pd.read_csv(os.path.join(path_to_AffectNet, "train_labels.csv"))
    AffectNet_dev = pd.read_csv(os.path.join(path_to_AffectNet, "dev_labels.csv"))

    # change filepaths to new ones
    AFEW_VA["frame_num"]=AFEW_VA["frame_num"].astype(str)
    RECOLA["filename"]=RECOLA["filename"].astype(str)
    SEMAINE["filename"]=SEMAINE["filename"].astype(str)
    SEWA["filename"]=SEWA["filename"].astype(str)
    AffectNet_train["abs_path"]=AffectNet_train["abs_path"].astype(str)
    AffectNet_dev["abs_path"]=AffectNet_dev["abs_path"].astype(str)

    AFEW_VA["frame_num"] = AFEW_VA["frame_num"].apply(lambda x: x.replace("/media/external_hdd_1/Datasets", "/work/home/dsu/Datasets"))
    RECOLA["filename"] = RECOLA["filename"].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    SEMAINE["filename"] = SEMAINE["filename"].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    SEWA["filename"] = SEWA["filename"].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    AffectNet_train["abs_path"] = AffectNet_train["abs_path"].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))
    AffectNet_dev["abs_path"] = AffectNet_dev["abs_path"].apply(lambda x: x.replace("/media/external_hdd_2/Datasets", "/work/home/dsu/Datasets"))

    # save dataframes
    AFEW_VA.to_csv(os.path.join(path_to_AFEW_VA, "labels.csv"), index=False)
    RECOLA.to_csv(os.path.join(path_to_RECOLA, "preprocessed_labels.csv"), index=False)
    SEMAINE.to_csv(os.path.join(path_to_SEMAINE, "preprocessed_labels.csv"), index=False)
    SEWA.to_csv(os.path.join(path_to_SEWA, "preprocessed_labels.csv"), index=False)
    AffectNet_train.to_csv(os.path.join(path_to_AffectNet, "train_labels.csv"), index=False)
    AffectNet_dev.to_csv(os.path.join(path_to_AffectNet, "dev_labels.csv"), index=False)"""





