import sys

from src.training.dynamic.facial.data_preparation import get_data_loaders

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])
from typing import Dict, Any

import numpy as np
import sklearn
import torch
import pandas as pd

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1


def create_and_load_feature_extractor(device:torch.device)->torch.nn.Module:
    path_to_weights_base_model = "/work/home/dsu/tmp/radiant_fog_160.pth"
    extractor = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8,
                                               num_regression_neurons=2)
    extractor.load_state_dict(torch.load(path_to_weights_base_model))
    # cut off last two layers of base model
    extractor = torch.nn.Sequential(*list(extractor.children())[:-2])
    # freeze base model
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    extractor.to(device)
    return extractor


def extract_features(extractor:torch.nn.Module, data_loader:torch.utils.data.DataLoader,
                     device:torch.device)->pd.DataFrame:
    # define column names
    column_names = ["emb_{}".format(i) for i in range(256)]+["arousal", "valence"]
    # create empty dataframe
    result = pd.DataFrame(columns=column_names)
    # extract features
    for i, (x, y) in enumerate(data_loader):
        # transform labels. We take the last value of each sequence as we need to predict only the last affective state
        y = y[:, -1, :]
        x = x.float()
        x = x.to(device)
        # Forward pass
        features = extractor(x)
        # append features to dataframe
        features = features.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        data_with_labels = np.concatenate((features, y), axis=1)
        data_with_labels = pd.DataFrame(data_with_labels, columns=column_names)
        result = result.append(data_with_labels)
    return result



def train_svm(svm_config:Dict[str, Any], train_data:pd.DataFrame)->sklearn.svm.SVR:
    svc = sklearn.svm.SVR(**svm_config)
    svc.fit(train_data.iloc[:, :-2], train_data.iloc[:, -2:])
    return svc

def evaluate_svm(model:sklearn.svm.SVR, data:pd.DataFrame)->Dict[str, float]:
    predictions = model.predict(data.iloc[:, :-2])
    results = {}
    results["arousal_mse"] = sklearn.metrics.mean_squared_error(data.iloc[:, -2], predictions[:, 0])
    results["valence_mse"] = sklearn.metrics.mean_squared_error(data.iloc[:, -1], predictions[:, 1])
    results["arousal_mae"] = sklearn.metrics.mean_absolute_error(data.iloc[:, -2], predictions[:, 0])
    results["valence_mae"] = sklearn.metrics.mean_absolute_error(data.iloc[:, -1], predictions[:, 1])
    results["arousal_rmse"] = results["arousal_mse"] ** 0.5
    results["valence_rmse"] = results["valence_mse"] ** 0.5
    return results



if __name__=="__main__":
    window_sizes = [1,2,3,4]
    svc_config_params = {"C": [0.1, 1, 10, 100],
                            "gamma": [1, 0.1, 0.01, 0.001],
                            "kernel": ["rbf", "linear", "poly", "sigmoid"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor=create_and_load_feature_extractor(device=device)
    results = pd.DataFrame(columns=["window_size", "svm_config", "test_arousal_mse", "test_valence_mse",
                                    "test_arousal_mae", "test_valence_mae", "test_arousal_rmse", "test_valence_rmse",
                                    "dev_arousal_mse", "dev_valence_mse", "dev_arousal_mae", "dev_valence_mae",
                                    "dev_arousal_rmse", "dev_valence_rmse"])

    for window_size in window_sizes:
        train_generator, dev_generator, test_generator = get_data_loaders(window_size=int(window_size),
                                                                      stride=int(window_size // 2),
                                                                      base_model_type="EfficientNet-B1",
                                                                      batch_size=16)
        train_data = extract_features(extractor=feature_extractor,
                                      data_loader=train_generator,
                                      device=device)
        dev_data = extract_features(extractor=feature_extractor,
                                        data_loader=dev_generator,
                                        device=device)
        test_data = extract_features(extractor=feature_extractor,
                                        data_loader=test_generator,
                                        device=device)
        # train svm with different parameters
        for C in svc_config_params["C"]:
            for gamma in svc_config_params["gamma"]:
                for kernel in svc_config_params["kernel"]:
                    svm_config = {"C": C, "gamma": gamma, "kernel": kernel}
                    svm_model = train_svm(svm_config=svm_config, train_data=train_data)
                    # evaluate svm
                    train_results = evaluate_svm(model=svm_model, data=train_data)
                    dev_results = evaluate_svm(model=svm_model, data=dev_data)
                    test_results = evaluate_svm(model=svm_model, data=test_data)
                    # save results in results dataframe
                    results = results.append({"window_size": window_size,
                                                "svm_config": svm_config,
                                                "test_arousal_mse": test_results["arousal_mse"],
                                                "test_valence_mse": test_results["valence_mse"],
                                                "test_arousal_mae": test_results["arousal_mae"],
                                                "test_valence_mae": test_results["valence_mae"],
                                                "test_arousal_rmse": test_results["arousal_rmse"],
                                                "test_valence_rmse": test_results["valence_rmse"],
                                                "dev_arousal_mse": dev_results["arousal_mse"],
                                                "dev_valence_mse": dev_results["valence_mse"],
                                                "dev_arousal_mae": dev_results["arousal_mae"],
                                                "dev_valence_mae": dev_results["valence_mae"],
                                                "dev_arousal_rmse": dev_results["arousal_rmse"],
                                                "dev_valence_rmse": dev_results["valence_rmse"]},
                                                 ignore_index=True)
                    results.to_csv("results.csv", index=False)
                    print("window_size: {}, svm_config: {}".format(window_size, svm_config))
                    print("train_results: {}".format(train_results))
                    print("dev_results: {}".format(dev_results))
                    print("test_results: {}".format(test_results))
                    print("------------------------------------------------------")

