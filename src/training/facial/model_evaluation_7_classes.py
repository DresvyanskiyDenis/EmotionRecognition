import gc
import sys
from functools import partial

sys.path.extend(["/work/home/dsu/simpleHigherHRNet/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])


from typing import Dict, Tuple, Optional

from src.training.facial import training_config
from src.training.facial.data_preparation import load_data_and_construct_dataloaders
import numpy as np

from visualization.ConfusionMatrixVisualization import plot_and_save_confusion_matrix


import os.path

import pandas as pd
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, \
    mean_absolute_error

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, Modified_EfficientNet_B4, Modified_ViT_B_16, \
    Modified_MobileNetV3_large


def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device, print_metrics:Optional[bool]=True) -> Tuple[
    Dict[object, float], ...]:
    evaluation_metrics_classification = {'val_accuracy_classification': accuracy_score,
                                         'val_precision_classification': partial(precision_score, average='macro'),
                                         'val_recall_classification': partial(recall_score, average='macro'),
                                         'val_f1_classification': partial(f1_score, average='macro')
                                         }

    evaluation_metric_arousal = {'val_arousal_rmse': mean_squared_error,
                                 'val_arousal_mae': mean_absolute_error
                                 }

    evaluation_metric_valence = {'val_valence_rmse': mean_squared_error,
                                 'val_valence_mae': mean_absolute_error
                                 }
    # create arrays for predictions and ground truth labels
    predictions_classifier, predictions_arousal, predictions_valence = [], [], []
    ground_truth_classifier, ground_truth_arousal, ground_truth_valence = [], [], []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # forward pass
            outputs = model(inputs)
            regression_output = [outputs[1][:, 0], outputs[1][:, 1]]
            classification_output = outputs[0]

            # transform classification output to fit labels
            classification_output = classification_output[...,:-1]  # do not take into account the contempt class, which is the last one
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)
            # transform regression output to fit labels
            regression_output = [regression_output[0].cpu().numpy().squeeze(),
                                 regression_output[1].cpu().numpy().squeeze()]

            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels[:, 2].cpu().numpy().squeeze()
            regression_ground_truth = [labels[:, 0].cpu().numpy().squeeze(), labels[:, 1].cpu().numpy().squeeze()]

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_arousal.append(regression_output[0])
            predictions_valence.append(regression_output[1])
            predictions_classifier.append(classification_output)
            ground_truth_arousal.append(regression_ground_truth[0])
            ground_truth_valence.append(regression_ground_truth[1])
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_arousal = np.concatenate(predictions_arousal, axis=0)
        predictions_valence = np.concatenate(predictions_valence, axis=0)
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_arousal = np.concatenate(ground_truth_arousal, axis=0)
        ground_truth_valence = np.concatenate(ground_truth_valence, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # create mask for all NaN values to remove them from evaluation
        mask_arousal = ~np.isnan(ground_truth_arousal)
        mask_valence = ~np.isnan(ground_truth_valence)
        mask_classifier = ~np.isnan(ground_truth_classifier)
        # remove NaN values from arrays
        predictions_arousal = predictions_arousal[mask_arousal]
        predictions_valence = predictions_valence[mask_valence]
        predictions_classifier = predictions_classifier[mask_classifier]
        ground_truth_arousal = ground_truth_arousal[mask_arousal]
        ground_truth_valence = ground_truth_valence[mask_valence]
        ground_truth_classifier = ground_truth_classifier[mask_classifier]

        # remove 7th class from classification
        mask_7th_class = ground_truth_classifier != 7
        predictions_classifier = predictions_classifier[mask_7th_class]
        ground_truth_classifier = ground_truth_classifier[mask_7th_class]


        # calculate evaluation metrics
        if len(ground_truth_arousal) != 0:
            evaluation_metrics_arousal = {
                metric: evaluation_metric_arousal[metric](ground_truth_arousal, predictions_arousal) for metric in
                evaluation_metric_arousal}
            evaluation_metrics_valence = {
                metric: evaluation_metric_valence[metric](ground_truth_valence, predictions_valence) for metric in
                evaluation_metric_valence}
        else:
            evaluation_metrics_arousal = {'val_arousal_rmse': np.NaN, 'val_arousal_mae': np.NaN}
            evaluation_metrics_valence = {'val_valence_rmse': np.NaN, 'val_valence_mae': np.NaN}
        evaluation_metrics_classifier = {
            metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier) for
            metric in evaluation_metrics_classification}
        # print evaluation metrics
        if print_metrics:
            print('Evaluation metrics for arousal:')
            for metric_name, metric_value in evaluation_metrics_arousal.items():
                print("%s: %.4f" % (metric_name, metric_value))
            print('Evaluation metrics for valence:')
            for metric_name, metric_value in evaluation_metrics_valence.items():
                print("%s: %.4f" % (metric_name, metric_value))
            print('Evaluation metrics for classifier:')
            for metric_name, metric_value in evaluation_metrics_classifier.items():
                print("%s: %.4f" % (metric_name, metric_value))
    # clear RAM from unused variables
    del inputs, labels, outputs, regression_output, classification_output, classification_ground_truth, \
        regression_ground_truth, mask_arousal, mask_valence, mask_classifier
    torch.cuda.empty_cache()
    return (evaluation_metrics_arousal, evaluation_metrics_valence, evaluation_metrics_classifier)


def draw_confusion_matrix(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device,
                          output_path:str, filename:str) -> None:

    # create arrays for predictions and ground truth labels
    predictions_classifier, predictions_arousal, predictions_valence = [], [], []
    ground_truth_classifier, ground_truth_arousal, ground_truth_valence = [], [], []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # forward pass
            outputs = model(inputs)
            regression_output = [outputs[1][:, 0], outputs[1][:, 1]]
            classification_output = outputs[0]

            # transform classification output to fit labels
            classification_output = classification_output[...,:-1]  # do not take into account the contempt class, which is the last one
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)
            # transform regression output to fit labels
            regression_output = [regression_output[0].cpu().numpy().squeeze(),
                                 regression_output[1].cpu().numpy().squeeze()]

            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels[:, 2].cpu().numpy().squeeze()
            regression_ground_truth = [labels[:, 0].cpu().numpy().squeeze(), labels[:, 1].cpu().numpy().squeeze()]

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_arousal.append(regression_output[0])
            predictions_valence.append(regression_output[1])
            predictions_classifier.append(classification_output)
            ground_truth_arousal.append(regression_ground_truth[0])
            ground_truth_valence.append(regression_ground_truth[1])
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_arousal = np.concatenate(predictions_arousal, axis=0)
        predictions_valence = np.concatenate(predictions_valence, axis=0)
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_arousal = np.concatenate(ground_truth_arousal, axis=0)
        ground_truth_valence = np.concatenate(ground_truth_valence, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # create mask for all NaN values to remove them from evaluation
        mask_arousal = ~np.isnan(ground_truth_arousal)
        mask_valence = ~np.isnan(ground_truth_valence)
        mask_classifier = ~np.isnan(ground_truth_classifier)
        # remove NaN values from arrays
        predictions_arousal = predictions_arousal[mask_arousal]
        predictions_valence = predictions_valence[mask_valence]
        predictions_classifier = predictions_classifier[mask_classifier]
        ground_truth_arousal = ground_truth_arousal[mask_arousal]
        ground_truth_valence = ground_truth_valence[mask_valence]
        ground_truth_classifier = ground_truth_classifier[mask_classifier]

        # remove 7th class from classification
        mask_7th_class = ground_truth_classifier != 7
        predictions_classifier = predictions_classifier[mask_7th_class]
        ground_truth_classifier = ground_truth_classifier[mask_7th_class]

        # draw confusion matrix for classification
        # draw confusion matrix
        plot_and_save_confusion_matrix(y_true=ground_truth_classifier, y_pred=predictions_classifier,
                                       name_labels=list(training_config.EMO_CATEGORIES.keys())[:-1], # do not take into account the contempt class, which is the last one
                                       path_to_save=output_path, name_filename=filename, title='Confusion matrix')

    # clear RAM from unused variables
    del inputs, labels, outputs, regression_output, classification_output, classification_ground_truth, \
        regression_ground_truth, mask_arousal, mask_valence, mask_classifier
    torch.cuda.empty_cache()


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


def test_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device, prefix:str) -> Tuple[
    Dict[str, float], ...]:
    test_metrics = evaluate_model(model, generator, device)
    # change the prefix of the metrics names from 'val_' to 'test_'
    test_metrics_arousal, test_metrics_valence, test_metrics_classification = test_metrics
    test_metrics_arousal = {key.replace('val_', prefix): value for key, value in test_metrics_arousal.items()}
    test_metrics_valence = {key.replace('val_', prefix): value for key, value in test_metrics_valence.items()}
    test_metrics_classification = {key.replace('val_', prefix): value for key, value in test_metrics_classification.items()}
    # pack the metrics back into the tuple
    test_metrics = (test_metrics_arousal, test_metrics_valence, test_metrics_classification)
    return test_metrics


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

    # test all models on the test set and dev set
    info['test_accuracy'] = -100
    info['test_precision'] = -100
    info['test_recall'] = -100
    info['test_f1'] = -100
    info['test_arousal_rmse'] = -100
    info['test_valence_rmse'] = -100
    info['dev_accuracy'] = -100
    info['dev_precision'] = -100
    info['dev_recall'] = -100
    info['dev_f1'] = -100
    info['dev_arousal_rmse'] = -100
    info['dev_valence_rmse'] = -100
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
        elif model_type == "MobileNetV3_large":
            model = Modified_MobileNetV3_large(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
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

        # validate model on dev set
        dev_metrics = test_model(model, dev_generator, device, prefix='dev_')
        # unpack metrics
        dev_metrics_arousal, dev_metrics_valence, dev_metrics_classification = dev_metrics
        # draw confusion matrix
        if not os.path.exists(os.path.join(output_path_for_models_weights, 'confusion_matrices')):
            os.makedirs(os.path.join(output_path_for_models_weights, 'confusion_matrices'))
        draw_confusion_matrix(model=model, generator=dev_generator, device=device,
                                output_path=os.path.join(output_path_for_models_weights, 'confusion_matrices'),
                                filename=info['ID'].iloc[i] + '_dev.png')
        # save dev metrics
        info.loc[i, 'dev_accuracy'] = dev_metrics_classification['dev_accuracy_classification']
        info.loc[i, 'dev_precision'] = dev_metrics_classification['dev_precision_classification']
        info.loc[i, 'dev_recall'] = dev_metrics_classification['dev_recall_classification']
        info.loc[i, 'dev_f1'] = dev_metrics_classification['dev_f1_classification']

        info.loc[i, 'dev_arousal_rmse'] = dev_metrics_arousal['dev_arousal_rmse']
        info.loc[i, 'dev_arousal_mae'] = dev_metrics_arousal['dev_arousal_mae']

        info.loc[i, 'dev_valence_rmse'] = dev_metrics_valence['dev_valence_rmse']
        info.loc[i, 'dev_valence_mae'] = dev_metrics_valence['dev_valence_mae']


        # test model
        test_metrics = test_model(model, test_generator, device, prefix='test_')
        # unpack metrics
        test_metrics_arousal, test_metrics_valence, test_metrics_classification = test_metrics
        # draw confusion matrix
        if not os.path.exists(os.path.join(output_path_for_models_weights, 'confusion_matrices')):
            os.makedirs(os.path.join(output_path_for_models_weights, 'confusion_matrices'))
        draw_confusion_matrix(model=model, generator=test_generator, device=device,
                              output_path=os.path.join(output_path_for_models_weights, 'confusion_matrices'),
                              filename=info['ID'].iloc[i] + '_test.png')


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
