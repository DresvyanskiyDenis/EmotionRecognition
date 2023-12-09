import os

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from src.training.dynamic.facial.data_preparation import load_one_dataset, separate_various_videos, \
    construct_data_loaders
from src.training.dynamic.facial.model_evaluation import evaluate_model, CCC
from src.training.dynamic.facial.models import Transformer_model_b1, GRU_model_b1, Simple_CNN


def get_info_and_download_models_weights_from_project(entity: str, project_name: str, output_path: str) -> pd.DataFrame:
    """ Extracts info about run models from the project and downloads the models weights to the output_path.
        The extracted information will be stored as pd.DataFrame with the columns:
        ['ID', 'base_model_type', 'architecture', 'sequence_length', 'val_RMSE_arousal', 'val_RMSE_valence',
        'val_MAE_arousal', 'val_MAE_valence', 'val_RMSE']

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
    info = pd.DataFrame(columns=['ID', 'base_model_type', 'architecture', 'sequence_length', 'val_RMSE_arousal',
                                 'val_RMSE_valence', 'val_MAE_arousal', 'val_MAE_valence', 'val_RMSE'])
    for run in runs:
        print('Downloading the model weights from the run: ', run.name)
        ID = run.name
        base_model_type = run.config['base_model_type']
        architecture = run.config['architecture']
        sequence_length = run.config['sequence_length']
        val_RMSE_arousal = run.summary['val_RMSE_arousal']
        val_RMSE_valence = run.summary['val_RMSE_valence']
        val_MAE_arousal = run.summary['val_MAE_arousal']
        val_MAE_valence = run.summary['val_MAE_valence']
        val_RMSE = run.summary['val_RMSE']
        # pack the info into the DataFrame
        info = pd.concat([info,
                          pd.DataFrame.from_dict(
                              {'ID': [ID], 'base_model_type': [base_model_type], 'architecture': [architecture],
                               'sequence_length': [sequence_length], 'val_RMSE_arousal': [val_RMSE_arousal],
                               'val_RMSE_valence': [val_RMSE_valence], 'val_MAE_arousal': [val_MAE_arousal],
                               'val_MAE_valence': [val_MAE_valence], 'val_RMSE': [val_RMSE]}
                          )
                          ]
                         )
        # download the model weights
        final_output_path = os.path.join(output_path, ID)
        run.file('best_model.pth').download(final_output_path, replace=True)
        # move the file out of dir and rename file for convenience
        os.rename(os.path.join(final_output_path, 'best_model.pth'),
                  final_output_path + '.pth')
        # delete the dir
        os.rmdir(final_output_path)
    return info


def get_data_loaders_for_separate_dataset(dataset_name: str, window_size: int, stride: int):
    # load data and separate it on videos
    train, dev, test = load_one_dataset(dataset_name)
    train, dev, test = separate_various_videos(train), separate_various_videos(dev), separate_various_videos(test)
    # construct data loaders
    train_dataloader, dev_dataloader, test_dataloader = construct_data_loaders(train, dev, test,
                                                                               window_size=window_size,
                                                                               stride=stride,
                                                                               preprocessing_functions=None,
                                                                               batch_size=1,
                                                                               augmentation_functions=None)
    return (train_dataloader, dev_dataloader, test_dataloader)


def main():
    # set the project name
    entity = 'denisdresvyanskiy'
    project_name = 'Emotion_Recognition_Seq2One'
    datasets = ['AFEW-VA', 'RECOLA', 'SEWA', 'SEMAINE']
    # set the output path
    output_path = '/work/home/dsu/emotion_recognition_project/weights_best_models/emotion_recognition_seq2one/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # metrics
    n_metrics = {'MSE_val_arousal': mean_squared_error,
                 'MAE_val_arousal': mean_absolute_error,
                 'MSE_val_valence': mean_squared_error,
                 'MAE_val_valence': mean_absolute_error,
                 }
    CCC_metrics = {'CCC_val_arousal': CCC,
                   'CCC_val_valence': CCC,
                   }
    # get the info about the models and download the models weights
    info = get_info_and_download_models_weights_from_project(entity=entity,
                                                             project_name=project_name,
                                                             output_path=output_path)
    # create new dataframe result_info where we will have the following columns
    # ID', 'base_model_type', 'architecture','sequence_length', 'dataset', 'val_RMSE_arousal', 'val_RMSE_valence',
    # 'val_MAE_arousal', 'val_MAE_valence', 'RMSE_dev_arousal', 'RMSE_dev_valence', 'CCC_dev_arousal', 'CCC_dev_valence'
    result_info = pd.DataFrame(columns=['ID', 'base_model_type', 'architecture', 'sequence_length', 'dataset',
                                        'RMSE_dev_arousal', 'RMSE_dev_valence', 'MAE_dev_arousal', 'MAE_dev_valence',
                                        'RMSE_test_arousal', 'RMSE_test_valence', 'MAE_test_arousal', 'MAE_test_valence',
                                        'CCC_dev_arousal', 'CCC_dev_valence', 'CCC_test_arousal', 'CCC_test_valence'])

    for i in tqdm(range(len(info))):
        print("Testing model %d / %s" % (i + 1, info['architecture'].iloc[i]))
        model_type = info['architecture'].iloc[i]
        sequence_length = info['sequence_length'].iloc[i]
        # create_model
        if model_type == "transformer":
            model = Transformer_model_b1(seq_len=sequence_length)
        elif model_type == "gru":
            model = GRU_model_b1(seq_len=sequence_length)
        elif model_type == "simple_cnn":
            model = Simple_CNN(seq_len=sequence_length)
        else:
            raise ValueError("Unknown model type: %s" % model_type)
        # load the model weights
        model.load_state_dict(torch.load(os.path.join(output_path, info['ID'].iloc[i] + '.pth')))
        model.eval()
        model.to(device)

        # go through all datasets
        for dataset_name in datasets:
            n_train_generator, n_dev_generator, n_test_generator = \
                get_data_loaders_for_separate_dataset(dataset_name=dataset_name,
                                                      window_size=info['sequence_length'].iloc[i],
                                                      stride=int(info['sequence_length'].iloc[i] * 0.4))
            ccc_train_generator, ccc_dev_generator, ccc_test_generator = \
                get_data_loaders_for_separate_dataset(dataset_name=dataset_name,
                                                      window_size=info['sequence_length'].iloc[i],
                                                      stride=1)
            # evaluate the model for getting RMSE and MAE
            n_dev_metrics = evaluate_model(model=model, data_generator=n_dev_generator,
                                           metrics=n_metrics, device=device)
            n_test_metrics = evaluate_model(model=model, data_generator=n_test_generator,
                                            metrics=n_metrics, device=device)
            # add to all metrics name the 'val_' or 'test_' prefix (depending on the data_generator)
            n_dev_metrics = {key.replace('val_', 'dev_'): value for key, value in n_dev_metrics.items()}
            n_test_metrics = {key.replace('val_', 'test_'): value for key, value in n_test_metrics.items()}

            # evaluate the model for getting CCC
            CCC_dev_metrics = evaluate_model(model=model, data_generator=ccc_dev_generator,
                                             metrics=CCC_metrics, device=device)
            CCC_test_metrics = evaluate_model(model=model, data_generator=ccc_test_generator,
                                              metrics=CCC_metrics, device=device)
            # add to all metrics name the 'val_' or 'test_' prefix (depending on the data_generator)
            CCC_dev_metrics = {key.replace('val_', 'dev_'): value for key, value in CCC_dev_metrics.items()}
            CCC_test_metrics = {key.replace('val_', 'test_'): value for key, value in CCC_test_metrics.items()}

            # put everything in the result_info dataframe
            new_row = pd.DataFrame.from_dict({'ID': [info['ID'].iloc[i]],
                                              'base_model_type': [info['base_model_type'].iloc[i]],
                                              'architecture': [info['architecture'].iloc[i]],
                                              'sequence_length': [info['sequence_length'].iloc[i]],
                                              'dataset': [dataset_name],
                                              'RMSE_dev_arousal': [n_dev_metrics['MSE_dev_arousal']**0.5],
                                              'RMSE_dev_valence': [n_dev_metrics['MSE_dev_valence']**0.5],
                                              'MAE_dev_arousal': [n_dev_metrics['MAE_dev_arousal']],
                                              'MAE_dev_valence': [n_dev_metrics['MAE_dev_valence']],
                                              'RMSE_test_arousal': [n_test_metrics['MSE_test_arousal']**0.5],
                                              'RMSE_test_valence': [n_test_metrics['MSE_test_valence']**0.5],
                                              'MAE_test_arousal': [n_test_metrics['MAE_test_arousal']],
                                              'MAE_test_valence': [n_test_metrics['MAE_test_valence']],
                                              'CCC_dev_arousal': [CCC_dev_metrics['CCC_dev_arousal']],
                                              'CCC_dev_valence': [CCC_dev_metrics['CCC_dev_valence']],
                                              'CCC_test_arousal': [CCC_test_metrics['CCC_test_arousal']],
                                              'CCC_test_valence': [CCC_test_metrics['CCC_test_valence']]
                                              })
            result_info = pd.concat([result_info, new_row], ignore_index=True)

        # save the result_info DataFrame
        result_info.to_csv(os.path.join(output_path, 'result_info_separate_datasets.csv'), index=False)


if __name__ == '__main__':
    main()
