from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import torch
from torch.nn.functional import one_hot

import training_config

def train_step(model:torch.nn.Module, optimizer:torch.optim.Optimizer, criterion:Tuple[torch.nn.Module,...],
               inputs:Tuple[torch.Tensor,...], ground_truths:List[torch.Tensor]) -> float:
    """ Performs one training step for a model.

    :param model: torch.nn.Module
            Model to train.
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterion: Tuple[torch.nn.Module,...]
            Loss functions for each output of the model.
    :param inputs: Tuple[torch.Tensor,...]
            Inputs for the model.
    :param ground_truths: Tuple[torch.Tensor,...]
            Ground truths for the model. Should be in the same order as the outputs of the model.
            Some elements can be NaN if the corresponding output is not used for training (label does not exist).
    :return:
    """
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = model(*inputs)
    total_loss = 0.
    # TODO: check if masking works right
    for i, criterion_i in enumerate(criterion):
        # calculate mask for current loss function
        mask = ~torch.isnan(ground_truths[i])
        # calculate loss based on mask
        loss = criterion_i(outputs[i][mask], ground_truths[i][mask])
        total_loss += loss.item()
        loss.backward()
    optimizer.step()
    return total_loss


def train_epoch(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer, criterions:Tuple[torch.nn.Module,...],
               device:torch.device, print_step:int=100)->float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterions: List[torch.nn.Module,...]
            Loss functions for each output of the model.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
    :return: float
            Average loss for the epoch.
    """

    running_loss = 0.0
    total_loss = 0.0
    counter = 0.0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float()
        inputs = inputs.to(device)

        # separate labels into a list of torch.Tensor
        labels = [labels[:,0], labels[:,1], labels[:,2]]
        labels[2] = one_hot(labels[2].long(), num_classes=training_config.NUM_CLASSES)
        labels = [label.float().to(device) for label in labels]

        # do train step
        step_loss = train_step(model, optimizer, criterions, inputs, labels)


        # print statistics
        running_loss += step_loss
        total_loss += step_loss
        counter += 1.
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
    return total_loss / counter


def train_model(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader, device:torch.device) -> None:
    pass


