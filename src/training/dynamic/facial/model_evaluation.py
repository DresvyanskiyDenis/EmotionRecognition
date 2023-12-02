import gc
import sys
from typing import Dict, Callable

import numpy as np
import torch.nn

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])


def CCC(y_true:np.ndarray, y_pred:np.ndarray)->float:
    """
    Computes the concordance correlation coefficient (CCC) for two arrays.
    :param y_true: np.ndarray
        The ground truth values.
    :param y_pred: np.ndarray
        The predicted values.
    :return: float
        The CCC value.
    """
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2 * covar / (true_var + pred_var + (pred_mean - true_mean) ** 2)

    return ccc



def evaluate_model(model:torch.nn.Module, data_generator:torch.utils.data.DataLoader,
                   metrics:Dict[str, Callable], device:torch.device)->Dict[str, float]:
    """
    Evaluates a model on a given dataset (going through all generator's instances) using the given metrics.
    :param model: torch.nn.Module
        The model for the evaluation.
    :param data_generator: torch.utils.data.DataLoader
        The data generator to use.
    :param metrics: Dict[str, Callable]
        The metrics to use. Can be several ones.
    :param device: torch.device
        The device to use.
    """
    results = {}
    # Set model to eval mode
    model.eval()
    # Iterate over the dataset
    ground_truths_arousal = []
    ground_truths_valence = []
    predictions_arousal = []
    predictions_valence = []
    with torch.no_grad():
        for input, labels in data_generator:
            # transform labels. We take the last value of each sequence as we need to predict only the last affective state
            labels = labels[:, -1, :]
            input = input.float()
            input = input.to(device)
            # Forward pass
            output = model(input)
            # separate outputs on arousal valence
            output_arousal = output[:, 0].squeeze().cpu().numpy()
            output_valence = output[:, 1].squeeze().cpu().numpy()
            if len(output_arousal.shape) == 0:
                output_arousal = np.expand_dims(output_arousal, axis=0)
                output_valence = np.expand_dims(output_valence, axis=0)
            # Save predictions
            predictions_arousal.extend(output_arousal)
            predictions_valence.extend(output_valence)
            # separate ground truth
            ground_truth_arousal = labels[:, 0].squeeze().cpu().numpy()
            ground_truth_valence = labels[:, 1].squeeze().cpu().numpy()
            if len(ground_truth_arousal.shape) == 0:
                ground_truth_arousal = np.expand_dims(ground_truth_arousal, axis=0)
                ground_truth_valence = np.expand_dims(ground_truth_valence, axis=0)
            # Save ground truth
            ground_truths_arousal.extend(ground_truth_arousal)
            ground_truths_valence.extend(ground_truth_valence)
    # Compute metrics
    for metric_name, metric in metrics.items():
        if 'arousal' in metric_name:
            predictions = predictions_arousal
            ground_truth = ground_truths_arousal
        elif 'valence' in metric_name:
            predictions = predictions_valence
            ground_truth = ground_truths_valence
        if 'MAE' in metric_name or 'MSE' in metric_name:
            # just calculate metric, it will work fine for mae and mse
            results[metric_name] = metric(ground_truth, predictions)
        else:
            raise ValueError(f"Unknown metric {metric_name}.")
    # Free memory
    del predictions, ground_truth, predictions_arousal, predictions_valence, ground_truths_arousal, ground_truths_valence
    gc.collect()
    return results



