import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])

# the script to evaluate the model on every dataset separately
import os

from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor
from src.training.static.facial import training_config
from src.training.static.facial.data_preparation import load_one_dataset
from src.training.static.facial.model_evaluation_7_classes import evaluate_model


def get_dataloaders(dataset_name:str)->Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Provides the dataloaders for the train, dev, and test sets of the specified dataset
    """
    train, dev, test = load_one_dataset(dataset_name, seed=training_config.splitting_seed)
    preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                               EfficientNet_image_preprocessor()]

    dev_dataloader = None
    test_dataloader = None

    if dev is not None:
        dev_data_loader = ImageDataLoader(paths_with_labels=dev, preprocessing_functions=preprocessing_functions,
                                      augmentation_functions=None, shuffle=False)
        dev_dataloader = DataLoader(dev_data_loader, batch_size=32, num_workers=8, shuffle=False)
    if test is not None:
        test_data_loader = ImageDataLoader(paths_with_labels=test, preprocessing_functions=preprocessing_functions,
                                       augmentation_functions=None, shuffle=False)
        test_dataloader = DataLoader(test_data_loader, batch_size=32, num_workers=8, shuffle=False)

    return dev_dataloader, test_dataloader

def evaluate_model_on_one_dataset(model:torch.nn.Module, dataset_name:str, device:torch.device):
    dev_dataloader, test_dataloader = get_dataloaders(dataset_name)
    # make metrics variable None in case there is no dev or test set
    dev_metrics_arousal, dev_metrics_valence, dev_metrics_classification, test_metrics_arousal, \
    test_metrics_valence, test_metrics_classification = None, None, None, None, None, None
    # evaluate the model on the dev and test sets
    if dev_dataloader is not None:
        dev_metrics_arousal, dev_metrics_valence, dev_metrics_classification = evaluate_model(model, dev_dataloader, device,
                                                                                  print_metrics=False)
        # change the metrics prefix from 'val_' to 'dev_'
        dev_metrics_arousal = {key.replace("val_", "dev_"): value for key, value in dev_metrics_arousal.items()}
        dev_metrics_valence = {key.replace("val_", "dev_"): value for key, value in dev_metrics_valence.items()}
        dev_metrics_classification = {key.replace("val_", "dev_"): value for key, value in dev_metrics_classification.items()}
    if test_dataloader is not None:
        test_metrics_arousal, test_metrics_valence, test_metrics_classification = evaluate_model(model, test_dataloader, device,
                                                                                       print_metrics=False)
        # change the metrics prefix from 'val_' to 'test_'
        test_metrics_arousal = {key.replace("val_", "test_"): value for key, value in test_metrics_arousal.items()}
        test_metrics_valence = {key.replace("val_", "test_"): value for key, value in test_metrics_valence.items()}
        test_metrics_classification = {key.replace("val_", "test_"): value for key, value in test_metrics_classification.items()}

    return dev_metrics_arousal, dev_metrics_valence, dev_metrics_classification, \
           test_metrics_arousal, test_metrics_valence, test_metrics_classification


def create_and_load_model()->torch.nn.Module:
    """
    Creates and loads the model.
    """
    path_to_weights = "/work/home/dsu/tmp/radiant_fog_160.pth"
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                     num_regression_neurons=training_config.NUM_REGRESSION_NEURONS)
    model.load_state_dict(torch.load(path_to_weights))
    return model



def main():
    results = pd.DataFrame(columns=["dataset", "dev_arousal_rmse", "dev_valence_rmse", "dev_recall", "dev_accuracy",
                                    "test_arousal_rmse", "test_valence_rmse", "test_recall", "test_accuracy"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_and_load_model()
    model.to(device)
    model.eval()
    # params
    datasets = ["AFEW-VA", "AffectNet", "RECOLA", "SEWA", "SEMAINE", "FER_plus", "RAF_DB", "EMOTIC", "ExpW", "SAVEE"]
    output_path = "/work/home/dsu/tmp"
    for dataset in datasets:
        dev_metrics_arousal, dev_metrics_valence, dev_metrics_classification, \
        test_metrics_arousal, test_metrics_valence, test_metrics_classification = evaluate_model_on_one_dataset(model, dataset, device)
        # append does not work anymore. Using concat instead
        new_result = pd.DataFrame.from_dict({"dataset": dataset,
                                    "dev_arousal_rmse": [dev_metrics_arousal["dev_arousal_rmse"]**0.5 if dev_metrics_arousal is not None else None],
                                    "dev_valence_rmse": [dev_metrics_valence["dev_valence_rmse"]**0.5 if dev_metrics_valence is not None else None],
                                    "dev_recall": [dev_metrics_classification["dev_recall_classification"] if dev_metrics_classification is not None else None],
                                    "dev_accuracy": [dev_metrics_classification["dev_accuracy_classification"] if dev_metrics_classification is not None else None],
                                    "test_arousal_rmse": [test_metrics_arousal["test_arousal_rmse"]**0.5 if test_metrics_arousal is not None else None],
                                    "test_valence_rmse": [test_metrics_valence["test_valence_rmse"]**0.5 if test_metrics_valence is not None else None],
                                    "test_recall": [test_metrics_classification["test_recall_classification"] if test_metrics_classification is not None else None],
                                    "test_accuracy": [test_metrics_classification["test_accuracy_classification"] if test_metrics_classification is not None else None]})
        results = pd.concat([results, new_result], ignore_index=True)
        # print the results as well
        print("----------------------------------------------")
        print(f"Dataset: {dataset}")
        print("Dev arousal RMSE: ", dev_metrics_arousal["dev_arousal_rmse"]**0.5 if dev_metrics_arousal is not None else None)
        print("Dev valence RMSE: ", dev_metrics_valence["dev_valence_rmse"]**0.5 if dev_metrics_valence is not None else None)
        print("Dev classification recall: ", dev_metrics_classification["dev_recall_classification"] if dev_metrics_classification is not None else None)
        print("Dev classification accuracy: ", dev_metrics_classification["dev_accuracy_classification"] if dev_metrics_classification is not None else None)
        print("Test arousal RMSE: ", test_metrics_arousal["test_arousal_rmse"]**0.5 if test_metrics_arousal is not None else None)
        print("Test valence RMSE: ", test_metrics_valence["test_valence_rmse"]**0.5 if test_metrics_valence is not None else None)
        print("Test classification recall: ", test_metrics_classification["test_recall_classification"] if test_metrics_classification is not None else None)
        print("Test classification accuracy: ", test_metrics_classification["test_accuracy_classification"] if test_metrics_classification is not None else None)
        print("---------------------------------------------")
        print()
    results.to_csv(os.path.join(output_path, "separate_evaluation_results.csv"), index=False)




if __name__=="__main__":
    main()
