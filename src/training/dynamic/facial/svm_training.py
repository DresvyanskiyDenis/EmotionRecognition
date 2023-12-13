import gc
import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])

from typing import Dict, Any, Tuple
from sklearnex import patch_sklearn
patch_sklearn()


import numpy as np
import sklearn
import torch
import pandas as pd
from sklearn.svm import SVR

from src.training.dynamic.facial.data_preparation import get_data_loaders
from src.training.dynamic.facial.model_evaluation import CCC



def extract_features(data_loader:torch.utils.data.DataLoader)->pd.DataFrame:
    # define column names
    column_names = ["emb_{}".format(i) for i in range(512)]+["arousal", "valence"]
    # create empty dataframe
    result = pd.DataFrame(columns=column_names)
    # extract features
    for i, (x, y) in enumerate(data_loader):
        # transform labels. We take the last value of each sequence as we need to predict only the last affective state
        y = y[:, -1, :]
        x = x.float()
        # append features to dataframe
        features = x.cpu().detach().numpy() # shape: (batch_size, seq_len, 256)
        # extract mean and std of each sequence
        features = np.concatenate((features.mean(axis=1), features.std(axis=1)), axis=1).squeeze() # shape: (batch_size, 512)
        y = y.cpu().detach().numpy().squeeze() # shape: (batch_size, 2)
        data_with_labels = np.concatenate((features, y), axis=1)
        data_with_labels = pd.DataFrame(data_with_labels, columns=column_names)
        result = pd.concat([result, data_with_labels], ignore_index=True)
    return result



def train_svm(svm_config:Dict[str, Any], train_data:pd.DataFrame)->Tuple[SVR, SVR]:
    svc_arousal = SVR(**svm_config)
    svc_valence = SVR(**svm_config)
    svc_arousal.fit(train_data.iloc[:, :-2], train_data.iloc[:, -2])
    svc_valence.fit(train_data.iloc[:, :-2], train_data.iloc[:, -1])
    return (svc_arousal, svc_valence)

def evaluate_svms(models:Tuple[SVR, SVR], data:pd.DataFrame)->Dict[str, float]:
    predictions_arousal = models[0].predict(data.iloc[:, :-2])
    predictions_valence = models[1].predict(data.iloc[:, :-2])
    results = {}
    results["arousal_mse"] = sklearn.metrics.mean_squared_error(data.iloc[:, -2], predictions_arousal.squeeze())
    results["valence_mse"] = sklearn.metrics.mean_squared_error(data.iloc[:, -1], predictions_valence.squeeze())
    results["arousal_mae"] = sklearn.metrics.mean_absolute_error(data.iloc[:, -2], predictions_arousal.squeeze())
    results["valence_mae"] = sklearn.metrics.mean_absolute_error(data.iloc[:, -1], predictions_valence.squeeze())
    results["arousal_rmse"] = results["arousal_mse"] ** 0.5
    results["valence_rmse"] = results["valence_mse"] ** 0.5
    results["CCC_arousal"] = CCC(data.iloc[:, -2], predictions_arousal.squeeze())
    results["CCC_valence"] = CCC(data.iloc[:, -1], predictions_valence.squeeze())
    return results



if __name__=="__main__":
    window_sizes = [5, 10, 15, 20]
    svc_config_params = {"C": [0.1, 1, 10, 100],
                            "gamma": [1, 0.1, 0.01, 0.001],
                            "kernel": ["rbf", "linear", "poly", "sigmoid"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = pd.DataFrame(columns=["window_size", "svm_config", "test_arousal_mse", "test_valence_mse",
                                    "test_arousal_mae", "test_valence_mae", "test_arousal_rmse", "test_valence_rmse",
                                    "dev_arousal_mse", "dev_valence_mse", "dev_arousal_mae", "dev_valence_mae",
                                    "dev_arousal_rmse", "dev_valence_rmse"])

    for window_size in window_sizes:
        train_generator, dev_generator, test_generator = get_data_loaders(window_size=int(window_size),
                                                                      stride=int(window_size // 2),
                                                                      base_model_type="EfficientNet-B1",
                                                                      batch_size=64)
        train_data = extract_features(data_loader=train_generator)
        dev_data = extract_features(data_loader=dev_generator)
        test_data = extract_features(data_loader=test_generator)
        # train svm with different parameters
        for C in svc_config_params["C"]:
            for gamma in svc_config_params["gamma"]:
                for kernel in svc_config_params["kernel"]:
                    print("Start training svm with window_size: {}, C: {}, gamma: {}, kernel: {}".format(window_size, C, gamma, kernel))
                    svm_config = {"C": C, "gamma": gamma, "kernel": kernel}
                    svm_models = train_svm(svm_config=svm_config, train_data=train_data)
                    print("Training finished.")
                    # evaluate svm
                    print("Start evaluation.")
                    train_results = evaluate_svms(models=svm_models, data=train_data)
                    dev_results = evaluate_svms(models=svm_models, data=dev_data)
                    test_results = evaluate_svms(models=svm_models, data=test_data)
                    print("Evaluation finished.")
                    # save results in results dataframe
                    new_record = pd.DataFrame.from_dict({"window_size": [window_size],
                                                "svm_config": [svm_config],
                                                "train_arousal_mse": [train_results["arousal_mse"]],
                                                "train_valence_mse": [train_results["valence_mse"]],
                                                "train_arousal_mae": [train_results["arousal_mae"]],
                                                "train_valence_mae": [train_results["valence_mae"]],
                                                "train_arousal_rmse": [train_results["arousal_rmse"]],
                                                "train_valence_rmse": [train_results["valence_rmse"]],
                                                "train_arousal_ccc": [train_results["CCC_arousal"]],
                                                "train_valence_ccc": [train_results["CCC_valence"]],
                                                "dev_arousal_mse": [dev_results["arousal_mse"]],
                                                "dev_valence_mse": [dev_results["valence_mse"]],
                                                "dev_arousal_mae": [dev_results["arousal_mae"]],
                                                "dev_valence_mae": [dev_results["valence_mae"]],
                                                "dev_arousal_rmse": [dev_results["arousal_rmse"]],
                                                "dev_valence_rmse": [dev_results["valence_rmse"]],
                                                "dev_arousal_ccc": [dev_results["CCC_arousal"]],
                                                "dev_valence_ccc": [dev_results["CCC_valence"]],
                                                "test_arousal_mse": [test_results["arousal_mse"]],
                                                "test_valence_mse": [test_results["valence_mse"]],
                                                "test_arousal_mae": [test_results["arousal_mae"]],
                                                "test_valence_mae": [test_results["valence_mae"]],
                                                "test_arousal_rmse": [test_results["arousal_rmse"]],
                                                "test_valence_rmse": [test_results["valence_rmse"]],
                                                "test_arousal_ccc": [test_results["CCC_arousal"]],
                                                "test_valence_ccc": [test_results["CCC_valence"]],
                                                })
                    results = pd.concat([results, new_record], ignore_index=True)
                    results.to_csv("/work/home/dsu/emotion_recognition_project/results.csv", index=False)
                    print("window_size: {}, svm_config: {}".format(window_size, svm_config))
                    print("train_results: {}".format(train_results))
                    print("dev_results: {}".format(dev_results))
                    print("test_results: {}".format(test_results))
                    print("------------------------------------------------------")
        # clear RAM
        del train_generator
        del dev_generator
        del test_generator
        del train_data
        del dev_data
        del test_data
        gc.collect()

