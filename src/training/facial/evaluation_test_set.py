import gc
import sys

import numpy as np

sys.path.extend(["/work/home/dsu/simpleHigherHRNet/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])


from typing import Dict, Tuple

from src.training.facial import training_config
from src.training.facial.data_preparation import load_data_and_construct_dataloaders
from src.training.facial.model_training_wandb import evaluate_model



import os.path

import pandas as pd
import torch
import wandb

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, Modified_EfficientNet_B4, Modified_ViT_B_16


def test_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device) -> Tuple[
    Dict[str, float], ...]:
    test_metrics = evaluate_model(model, generator, device)
    # change the prefix of the metrics names from 'val_' to 'test_'
    test_metrics_arousal, test_metrics_valence, test_metrics_classification = test_metrics
    test_metrics_arousal = {key.replace('val_', 'test_'): value for key, value in test_metrics_arousal.items()}
    test_metrics_valence = {key.replace('val_', 'test_'): value for key, value in test_metrics_valence.items()}
    test_metrics_classification = {key.replace('val_', 'test_'): value for key, value in test_metrics_classification.items()}
    # pack the metrics back into the tuple
    test_metrics = (test_metrics_arousal, test_metrics_valence, test_metrics_classification)
    return test_metrics


def get_info_and_download_models_weights_from_project(entity: str, project_name: str, output_path: str) -> pd.DataFrame:
    """ Extracts info about run models from the project and downloads the models weights to the output_path.
        The extracted information will be stored as pd.DataFrame with the columns:
        ['ID', 'model_type', 'discriminative_learning', 'gradual_unfreezing', 'loss_multiplication_factor', 'best_val_recall']

    :param entity: str
            The name of the WandB entity. (usually account name)
    :param project_name: str
            The name of the WandB project.
    :param output_path: str
            The path to the folder where the models weights will be downloaded.
    :return: pd.DataFrame
            The extracted information about the models.
    """
    # get api
    api = wandb.Api()
    # establish the entity and project name
    entity, project = entity, project_name
    # get runs from the project
    runs = api.runs(f"{entity}/{project}")
    # extract info about the runs
    info = pd.DataFrame(columns=['ID', 'model_type', 'discriminative_learning', 'gradual_unfreezing',
                                 'loss_multiplication_factor', 'best_val_general_metric', 'best_val_recall',
                                 'best_val_rmse_arousal', 'best_val_rmse_valence'])
    for run in runs:
        print('Downloading the model weights from the run: ', run.name)
        ID = run.name
        model_type = run.config['MODEL_TYPE']
        discriminative_learning = run.config['DISCRIMINATIVE_LEARNING']
        gradual_unfreezing = run.config['GRADUAL_UNFREEZING']
        loss_multiplication_factor = run.config['loss_multiplication_factor'] if 'loss_multiplication_factor' in list(run.config.keys()) else np.NaN
        best_val_general_metric = run.config['best_val_metric_value\n(Average sum of (1.-RMSE_arousal), (1.-RMSE_valence), and RECALL_classification)']
        best_val_recall = run.config['best_val_recall_classification']
        best_val_rmse_arousal = run.config['best_val_rmse_arousal']
        best_val_rmse_valence = run.config['best_val_rmse_valence']
        # pack the info into the DataFrame
        info = pd.concat([info,
                          pd.DataFrame.from_dict(
                              {'ID': [ID], 'model_type': [model_type],
                               'discriminative_learning': [discriminative_learning],
                               'gradual_unfreezing': [gradual_unfreezing],
                               'loss_multiplication_factor': [loss_multiplication_factor],
                               'best_val_general_metric': [best_val_general_metric],
                               'best_val_recall': [best_val_recall],
                               'best_val_rmse_arousal': [best_val_rmse_arousal],
                                'best_val_rmse_valence': [best_val_rmse_valence]}
                          )
                          ]
                         )
        # download the model weights
        final_output_path = os.path.join(output_path, ID)
        run.file('best_model_metric.pth').download(final_output_path, replace=True)
        # move the file out of dir and rename file for convenience
        os.rename(os.path.join(final_output_path, 'best_model_metric.pth'),
                  final_output_path + '.pth')
        # delete the dir
        os.rmdir(final_output_path)

    return info


if __name__ == "__main__":
    # params
    batch_size = 32
    project_name = 'Emotion_recognition_F2F'
    entity = 'denisdresvyanskiy'
    output_path_for_models_weights = "/" + os.path.join(*os.path.abspath(__file__).split(os.path.sep)[:-4],
                                                        'weights_best_models/')

    if not os.path.exists(output_path_for_models_weights):
        os.makedirs(output_path_for_models_weights)

    # get info about all runs (training sessions) and download the models weights
    info = get_info_and_download_models_weights_from_project(entity=entity, project_name=project_name,
                                                             output_path=output_path_for_models_weights)

    # test all models on the test set
    info['test_accuracy'] = -100
    info['test_precision'] = -100
    info['test_recall'] = -100
    info['test_f1'] = -100
    info.reset_index(drop=True, inplace=True)
    for i in range(len(info)):
        print("Testing model %d / %s" % (i + 1, info['model_type'].iloc[i]))
        # create model
        model_type = info['model_type'].iloc[i]
        if model_type == "EfficientNet-B1":
            model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                             num_regression_neurons=training_config.NUM_REGRESSION_NEURONS)
        elif model_type == "EfficientNet-B4":
            model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                             num_regression_neurons=training_config.NUM_REGRESSION_NEURONS)
        elif model_type == "ViT_B_16":
            model = Modified_ViT_B_16(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                      num_regression_neurons=training_config.NUM_REGRESSION_NEURONS)
        else:
            raise ValueError("Unknown model type: %s" % model_type)
        # load model weights
        path_to_weights = os.path.join(output_path_for_models_weights, info['ID'].iloc[i] + '.pth')
        model.load_state_dict(torch.load(path_to_weights))
        # define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # get generators, including test generator
        (train_generator, dev_generator, test_generator), class_weights = load_data_and_construct_dataloaders(
            model_type=model_type,
            batch_size=batch_size,
            return_class_weights=True)

        # test model
        test_metrics = test_model(model, test_generator, device)
        # unpack metrics
        test_metrics_arousal, test_metrics_valence, test_metrics_classification = test_metrics


        # save test metrics
        info.loc[i, 'test_accuracy'] = test_metrics_classification['test_accuracy_classification']
        info.loc[i, 'test_precision'] = test_metrics_classification['test_precision_classification']
        info.loc[i, 'test_recall'] = test_metrics_classification['test_recall_classification']
        info.loc[i, 'test_f1'] = test_metrics_classification['test_f1_classification']

        info.loc[i, 'test_arousal_rmse'] = test_metrics_arousal['test_arousal_rmse']
        info.loc[i, 'test_arousal_mae'] = test_metrics_arousal['test_arousal_mae']

        info.loc[i, 'test_valence_rmse'] = test_metrics_valence['test_valence_rmse']
        info.loc[i, 'test_valence_mae'] = test_metrics_valence['test_valence_mae']

        # save info
        info.to_csv(os.path.join(output_path_for_models_weights, 'info.csv'), index=False)

        # clear RAM and GPU memory
        del model, train_generator, dev_generator, test_generator, class_weights
        torch.cuda.empty_cache()
        gc.collect()
