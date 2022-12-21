from functools import partial
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.nn.functional import one_hot
from torchinfo import summary

import training_config
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping, TorchMetricEvaluator
from pytorch_utils.training_utils.losses import SoftFocalLoss
from src.models.ViT_models import ViT_Deit_model


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


def train_model(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader, dev_generator:torch.utils.data.DataLoader,
                device:torch.device) -> None:

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT_Deit_model(num_classes=training_config.NUM_CLASSES)
    model = model.to(device)
    summary(model, input_size=(config.batch_size, 3, 256, 256))
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[training_config.OPTIMIZER](model.parameters(), lr=training_config.LR_MAX)
    # Loss functions
    criterions = (torch.nn.MSELoss(), torch.nn.MSELoss(), SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2))
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.annealing_period,
                                                             eta_min=config.learning_rate_min),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
    }
    lr_scheduller = lr_schedullers[training_config.LR_SCHEDULLER]
    # callbacks
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score': partial(f1_score, average='macro')
    }
    best_val_recall = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=15,
                                                 save_path='best_model/',
                                                 mode="min")

    # TODO: write an evaluation loop



























if __name__ == "__main__":

    # initialization of Weights and Biases
    wandb.init(project="VGGFace2_FtF_training", config=metaparams)
    config = wandb.config

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HRNet = load_HRNet_model(device="cuda" if torch.cuda.is_available() else "cpu",
                             path_to_weights="/work/home/dsu/simpleHRNet/pose_hrnet_w32_256x192.pth")
    model = modified_HRNet(HRNet, num_classes=config.num_classes)
    model.to(device)
    summary(model, input_size=(config.batch_size, 3, 256, 256))
    # Select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}

    optimizer = optimizers[config.optimizer](model.parameters(), lr=config.learning_rate_max)
    # select loss function
    class_weights = torch.from_numpy(class_weights).float()
    class_weights = class_weights.to(device)
    criterions = {'Crossentropy': torch.nn.CrossEntropyLoss(weight=class_weights),
                  'Focal_loss': SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2)}
    criterion = criterions[loss_function]
    wandb.config.update({'loss': criterion})
    # Select lr scheduller
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.annealing_period,
                                                             eta_min=config.learning_rate_min),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
    }
    lr_scheduller = lr_schedullers[config.lr_scheduller]
    # callbacks
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score': partial(f1_score, average='macro')
    }
    best_val_recall = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=15,
                                                 save_path=wandb.run.dir,
                                                 mode="max")

    metric_evaluator = TorchMetricEvaluator(generator=dev,
                                            model=model,
                                            metrics=val_metrics,
                                            device=device,
                                            output_argmax=True,
                                            output_softmax=True,
                                            labels_argmax=True,
                                            loss_func=criterion)
    # print information about run
    print('Metaparams: optimizer: %s, learning_rate:%f, lr_scheduller: %s, annealing_period: %d, epochs: %d, '
          'batch_size: %d, augmentation_rate: %f, architecture: %s, '
          'dataset: %s, num_classes: %d' % (config.optimizer, config.learning_rate_max, config.lr_scheduller,
                                            config.annealing_period, config.epochs, config.batch_size,
                                            config.augmentation_rate, config.architecture, config.dataset,
                                            config.num_classes))
    # go through epochs
    for epoch in range(epochs):
        # train model one epoch
        loss = train_step(model=model, train_generator=train, optimizer=optimizer, criterion=criterion,
                          device=device, print_step=100)
        loss = loss / config.batch_size

        # evaluate model on dev set
        print('model evaluation...')
        with torch.no_grad():
            dev_results = metric_evaluator()
            print("Epoch: %i, dev results:" % epoch)
            for metric_name, metric_value in dev_results.items():
                print("%s: %.4f" % (metric_name, metric_value))
            # check early stopping
            early_stopping_result = early_stopping_callback(dev_results['val_recall'], model)
            # check if we have new best recall result on the validation set
            if dev_results['val_recall'] > best_val_recall:
                best_val_recall = dev_results['val_recall']
                wandb.config.update({'best_val_recall': best_val_recall}, allow_val_change=True)
        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(dev_results, commit=False)
        wandb.log({'train_loss': loss})
        # update lr
        if config.lr_scheduller == "ReduceLRonPlateau":
            lr_scheduller.step(dev_results['loss'])
        elif config.lr_scheduller == "Cyclic":
            lr_scheduller.step()
        # break the training loop if the model is not improving for a while
        if early_stopping_result:
            break
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()

