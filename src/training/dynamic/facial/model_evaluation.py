import gc
import sys
from typing import Dict, Callable

import torch.nn

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])


def evaluate_model(model:torch.nn.Module, data_generator:torch.utils.data.DataLoader,
                   metrics:Dict[str, Callable])->Dict[str, float]:
    """
    Evaluates a model on a given dataset (going through all generator's instances) using the given metrics.
    :param model: torch.nn.Module
        The model for the evaluation.
    :param data_generator: torch.utils.data.DataLoader
        The data generator to use.
    :param metrics: Dict[str, Callable]
        The metrics to use. Can be several ones.
    """
    results = {}
    # Set model to eval mode
    model.eval()
    # Iterate over the dataset
    ground_truth_arousal = []
    ground_truth_valence = []
    predictions_arousal = []
    predictions_valence = []
    with torch.no_grad():
        for input, labels in data_generator:
            # Forward pass
            output = model(input)
            # separate outputs on arousal valence
            output_arousal = output[:, :, 0].squeeze()
            output_valence = output[:, :, 1].squeeze()
            # Save predictions
            predictions_arousal.extend(output_arousal)
            predictions_valence.extend(output_valence)
            # separate ground truth
            ground_truth_arousal = labels[:, :, 0].squeeze()
            ground_truth_valence = labels[:, :, 1].squeeze()
            # Save ground truth
            ground_truth_arousal.extend(ground_truth_arousal)
            ground_truth_valence.extend(ground_truth_valence)
    # Compute metrics
    for metric_name, metric in metrics.items():
        if 'arousal' in metric_name:
            predictions = predictions_arousal
            ground_truth = ground_truth_arousal
        elif 'valence' in metric_name:
            predictions = predictions_valence
            ground_truth = ground_truth_valence
        results[metric_name] = metric(ground_truth, predictions)
    # TODO: check if the evaluation function works correct
    # Free memory
    del predictions, ground_truth
    gc.collect()
    return results



