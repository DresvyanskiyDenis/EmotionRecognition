import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])


import os

import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from src.training.dynamic.facial.data_preparation import get_data_loaders
from src.training.dynamic.facial.model_evaluation import evaluate_model
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
        run.file('best_model.pt').download(final_output_path, replace=True)
        # move the file out of dir and rename file for convenience
        os.rename(os.path.join(final_output_path, 'best_model.pt'),
                  final_output_path + '.pt')
        # delete the dir
        os.rmdir(final_output_path)
    return info



def main():

    # set the project name
    entity = 'denisdresvyanskiy'
    project_name = 'Emotion_Recognition_Seq2One'
    # set the output path
    output_path = '/work/home/dsu/emotion_recognition_project/weights_best_models/emotion_recognition_seq2one/'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # metrics
    metrics ={'MSE_val_arousal': mean_squared_error,
         'MAE_val_arousal': mean_absolute_error,
         'MSE_val_valence': mean_squared_error,
         'MAE_val_valence': mean_absolute_error,
         }
    # get the info about the models and download the models weights
    info = get_info_and_download_models_weights_from_project(entity=entity,
                                                                project_name=project_name,
                                                                output_path=output_path)
    # create dev and test columsn for the metrics
    info['dev_RMSE_arousal'] = None
    info['dev_RMSE_valence'] = None
    info['dev_MAE_arousal'] = None
    info['dev_MAE_valence'] = None
    info['dev_RMSE'] = None
    info['test_RMSE_arousal'] = None
    info['test_RMSE_valence'] = None
    info['test_MAE_arousal'] = None
    info['test_MAE_valence'] = None
    info['test_RMSE'] = None


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
        model.load_state_dict(torch.load(os.path.join(output_path, info['ID'].iloc[i] + '.pt')))
        train_generator, dev_generator, test_generator = get_data_loaders(window_size=info['sequence_length'].iloc[i],
                                                                      stride=int(info['sequence_length'].iloc[i]*0.4),
                                                                      base_model_type="EfficientNet-B1",
                                                                      batch_size=32)
        # evaluate the model
        model.eval()
        model.to(device)
        dev_metrics = evaluate_model(model=model, data_generator=dev_generator,
        metrics=metrics, device=device)
        test_metrics = evaluate_model(model=model, data_generator=test_generator,
                                      metrics = metrics, device=device)
        # add to all metrics name the 'val_' or 'test_' prefix (depending on the data_generator)
        dev_metrics = {key.replace('val_', 'dev_'): value for key, value in dev_metrics.items()}
        test_metrics = {key.replace('val_','test_'): value for key, value in test_metrics.items()}
        # add the metrics to the info DataFrame
        info['dev_RMSE_arousal'].iloc[i] = dev_metrics['MSE_dev_arousal']**0.5 # take the square root of the MSE
        info['dev_RMSE_valence'].iloc[i] = dev_metrics['MSE_dev_valence']**0.5
        info['dev_MAE_arousal'].iloc[i] = dev_metrics['MAE_dev_arousal']
        info['dev_MAE_valence'].iloc[i] = dev_metrics['MAE_dev_valence']
        info['dev_RMSE'].iloc[i] = (info['dev_RMSE_arousal'].iloc[i] + info['dev_RMSE_valence'].iloc[i])/2.
        info['test_RMSE_arousal'].iloc[i] = test_metrics['MSE_test_arousal']**0.5
        info['test_RMSE_valence'].iloc[i] = test_metrics['MSE_test_valence']**0.5
        info['test_MAE_arousal'].iloc[i] = test_metrics['MAE_test_arousal']
        info['test_MAE_valence'].iloc[i] = test_metrics['MAE_test_valence']
        info['test_RMSE'].iloc[i] = (info['test_RMSE_arousal'].iloc[i] + info['test_RMSE_valence'].iloc[i])/2.


        # save the info DataFrame
        info.to_csv(os.path.join(output_path, 'evaluation_all_datasets.csv'), index=False)






if __name__=="__main__":
    main()