import sys

from torchmetrics import ConcordanceCorrCoef

from src.training.dynamic.facial.model_evaluation import evaluate_model
from src.training.dynamic.facial.models import Transformer_model_b1, GRU_model_b1

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])



import argparse
from torchinfo import summary
import gc
import os
from functools import partial
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping
from src.training.dynamic.facial.data_preparation import get_data_loaders
from pytorch_utils.training_utils.losses import CCCLoss

import wandb


def train_step(model: torch.nn.Module, criterion: torch.nn.Module,
               input: torch.Tensor, ground_truth: torch.Tensor,
               device: torch.device) -> torch.Tensor:
    """ Performs one training step for a model.

    :param model: torch.nn.Module
            Model to train.
    :param criterion: torch.nn.Module
            Loss functions for output of the model.
    :param input: torch.Tensor
            Input for the model.
    :param ground_truth: torch.Tensor
            Ground truth for the model.
    :param device: torch.device
            Device to use for training.
    :return:
    """
    # forward pass
    output = model(input)

    # calculate criterion
    ground_truth = ground_truth.to(device)
    loss = criterion(output, ground_truth)

    # clear RAM from unused variables
    del output, ground_truth

    return loss


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device, print_step: int = 100,
                accumulate_gradients: Optional[int] = 1,
                warmup_lr_scheduller: Optional[object] = None) -> float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterion: torch.nn.Module
            Loss function for output of the model.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
    :param accumulate_gradients: Optional[int]
            Number of mini-batches to accumulate gradients for. If 1, no accumulation is performed.
    :param warmup_lr_scheduller: Optional[torch.optim.lr_scheduler]
            Learning rate scheduller in case we have warmup lr scheduller. In that case, the learning rate is being changed
            after every mini-batch, therefore should be passed to this function.
    :return: float
            Average loss for the epoch.
    """

    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float()
        inputs = inputs.to(device)

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            loss = train_step(model, criterion, inputs, labels, device)
            # normalize losses by number of accumulate gradient steps
            loss = loss / accumulate_gradients
            # backward pass
            loss.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_lr_scheduller is not None:
                    warmup_lr_scheduller.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        counter += 1
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
        # clear RAM from all the intermediate variables
        del inputs, labels, loss
    # clear RAM at the end of the epoch
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / counter


def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader, seq2seq_model_type:str,
                base_model_type:str, BATCH_SIZE:int, ACCUMULATE_GRADIENTS:int) -> None:
    # metaparams
    metaparams = {
        # general params
        "architecture": seq2seq_model_type,
        "base_model_type": base_model_type,
        "path_to_weights_base_model": "/work/home/dsu/tmp/radiant_fog_160.pth",
        "dataset": "RECOLA, SEWA, SEMAINE",
        "BEST_MODEL_SAVE_PATH": "best_models/",
        "NUM_WORKERS": 8,
        # training metaparams
        "NUM_EPOCHS": 100,
        "BATCH_SIZE": BATCH_SIZE,
        "OPTIMIZER": "AdamW",
        "AUGMENT_PROB": 0.03,
        "EARLY_STOPPING_PATIENCE": 50,
        "WEIGHT_DECAY": 0.0001,
        # LR scheduller params
        "LR_SCHEDULLER": "Warmup_cyclic",
        "ANNEALING_PERIOD": 5,
        "LR_MAX_CYCLIC": 0.005,
        "LR_MIN_CYCLIC": 0.0001,
        "LR_MIN_WARMUP": 0.00001,
        "WARMUP_STEPS": 100,
        "WARMUP_MODE": "linear",
    }
    print("____________________________________________________")
    print("Training params:")
    for key, value in metaparams.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="Emotion_Recognition_Seq2Seq", config=metaparams)
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH':wandb.run.dir}, allow_val_change=True)

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.base_model_type == "EfficientNet-B1":
        if config.architecture == "transformer":
            model = Transformer_model_b1(path_to_weights_base_model=config.path_to_weights_base_model)
        elif config.architecture == "gru":
            model = GRU_model_b1(path_to_weights_base_model=config.path_to_weights_base_model)
        else:
            raise ValueError("Unknown model type: %s" % config.architecture)
    else:
        raise ValueError("Unknown model type: %s" % config.base_model_type)
    model = model.to(device)

    # define all model layers (params), which will be used by optimizer
    model_parameters = model.parameters()
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[config.OPTIMIZER](model_parameters, lr=config.LR_MAX_CYCLIC,
                                             weight_decay=config.WEIGHT_DECAY)
    # print model summary
    print(summary(model, input_size=(config.BATCH_SIZE, 9, 3, 224, 224), verbose=0))
    # Loss functions
    criterion = CCCLoss(reduction='mean')  # TODO: check how is it working - does it take into account batch size?
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,
                                                             eta_min=0.0001),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=5,
                                                                                                 eta_min=0.0001),
                                         len_loader=len(train_generator),
                                         warmup_steps=100,
                                         warmup_start_lr=0.00001,
                                         warmup_mode="linear")
    }
    # if we use discriminative learning, we don't need LR scheduler
    lr_scheduller = lr_schedullers[config.LR_SCHEDULLER]
    # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
    if config.LR_SCHEDULLER == 'Warmup_cyclic':
        optimizer.param_groups[0]['lr'] = config.LR_MIN_WARMUP

    # evaluation aprams
    best_val_CCC = 0
    evaluation_metrics = {'MSE_val': mean_squared_error,
                            'MAE_val': mean_absolute_error,
                            'CCC_val': ConcordanceCorrCoef}
    # early stopping
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="max")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterion, device, print_step=100,
                                 accumulate_gradients=ACCUMULATE_GRADIENTS,
                                 warmup_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics = evaluate_model(model, dev_generator, evaluation_metrics)
        val_CCC = val_metrics['CCC_val']

        # update best val metrics got on validation set and log them using wandb
        if val_CCC > best_val_CCC:
            best_val_CCC = val_CCC
            wandb.config.update({'best_val_CCC': best_val_CCC}, allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model.pth'))
        print("Development CCC:" % val_CCC)

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log({'val_MSE': val_metrics['MSE_val']}, commit=False)
        wandb.log({'val_MAE': val_metrics['MAE_val']}, commit=False)
        wandb.log({'val_CCC': val_CCC}, commit=False)
        wandb.log({'train_loss_CCC': train_loss})
        # update LR if needed
        if config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(val_CCC)
        elif config.LR_SCHEDULLER == 'Cyclic':
            lr_scheduller.step()
        # check early stopping
        early_stopping_result = early_stopping_callback(val_CCC, model)
        if early_stopping_result:
            print("Early stopping")
            break
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(window_size, stride, base_model_type, seq2seq_model_type, batch_size, accumulate_gradients):
    print("Start of the script....")
    # get data loaders
    (train_generator, dev_generator, test_generator), class_weights = get_data_loaders(window_size=window_size,
                                                                                       stride=stride,
                                                                                       base_model_type=base_model_type,
                                                                                       batch_size=batch_size)
    # train the model
    train_model(train_generator=train_generator, dev_generator=dev_generator,
                base_model_type=base_model_type, seq2seq_model_type = seq2seq_model_type,
                BATCH_SIZE=batch_size, ACCUMULATE_GRADIENTS=accumulate_gradients)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Emotion Recognition model training',
        epilog='Parameters: model_type, batch_size, accumulate_gradients')
    parser.add_argument('--window_size', type=float, required=True)
    parser.add_argument('--stride', type=float, required=True)
    parser.add_argument('--base_model_type', type=str, required=True)
    parser.add_argument('--seq2seq_model_type', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--accumulate_gradients', type=int, required=True)
    args = parser.parse_args()
    # turn passed args from int to bool
    print("Passed args: ", args)
    # check arguments
    if args.base_model_type not in ['EfficientNet-B1', 'EfficientNet-B4']:
        raise ValueError("model_type should be either EfficientNet-B1 or EfficientNet-B4. Got %s" % args.model_type)
    if args.batch_size < 1:
        raise ValueError("batch_size should be greater than 0")
    if args.accumulate_gradients < 1:
        raise ValueError("accumulate_gradients should be greater than 0")
    # convert args to bool
    window_size = args.window_size
    stride = args.stride
    base_model_type = args.base_model_type
    seq2seq_model_type = args.seq2seq_model_type
    batch_size = args.batch_size
    accumulate_gradients = args.accumulate_gradients
    # run main script with passed args
    main(window_size=window_size, stride=stride,
         base_model_type = base_model_type, seq2seq_model_type = seq2seq_model_type,
         batch_size=batch_size, accumulate_gradients=accumulate_gradients)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()

