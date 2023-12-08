import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])



import argparse
from torchinfo import summary
import gc
import os
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping
from src.training.dynamic.facial.data_preparation import get_data_loaders
from pytorch_utils.training_utils.losses import RMSELoss
from src.training.dynamic.facial.model_evaluation import evaluate_model
from src.training.dynamic.facial.models import Transformer_model_b1, GRU_model_b1, Simple_CNN

import wandb


def train_step(model: torch.nn.Module, criterions: torch.nn.Module,
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
    # ground_truth shape: (batch_size, 2) Seq2One
    # output shape: (batch_size, 2)
    # forward pass
    output = model(input)
    # separate outputs on arousal valence
    output_arousal = output[:, 0].squeeze()
    output_valence = output[:, 1].squeeze()

    # separate ground truth
    ground_truth = ground_truth.to(device)
    ground_truth_arousal = ground_truth[:, 0].squeeze()
    ground_truth_valence = ground_truth[:, 1].squeeze()

    # calculate criterion
    loss_arousal = criterions[0](output_arousal, ground_truth_arousal)
    loss_valence = criterions[1](output_valence, ground_truth_valence)

    # calculate total loss
    loss = loss_arousal + loss_valence

    # clear RAM from unused variables
    del output, ground_truth, output_arousal, output_valence, ground_truth_arousal, ground_truth_valence

    return loss


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterions: torch.nn.Module,
                device: torch.device, print_step: int = 20,
                warmup_lr_scheduller: Optional[object] = None) -> float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterions: torch.nn.Module
            Loss functions for output of the model.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
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
        # transform labels. We take the last value of each sequence as we need to predict only the last affective state
        labels = labels[:, -1, :]

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            loss = train_step(model, criterions, inputs, labels, device)
            # backward pass
            loss.backward()
            # update weights
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



def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader, seq2one_model_type:str,
                base_model_type:str, BATCH_SIZE:int) -> None:
    # get the sequence length
    seq_len = next(iter(train_generator))[0].shape[1] # the input data shape: (batch_size, seq_len, channels, height, width)


    # metaparams
    metaparams = {
        # general params
        "architecture": seq2one_model_type,
        "base_model_type": base_model_type,
        "sequence_length": seq_len,
        "path_to_weights_base_model": "/work/home/dsu/tmp/radiant_fog_160.pth",
        "dataset": "RECOLA, SEWA, SEMAINE, AFEW-VA",
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
        "LR_MAX_CYCLIC": 0.01,
        "LR_MIN_CYCLIC": 0.001,
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
    wandb.init(project="Emotion_Recognition_Seq2One", config=metaparams)
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH':wandb.run.dir}, allow_val_change=True)

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get the sequence length
    if config.architecture == "transformer":
        model = Transformer_model_b1(seq_len=config.sequence_length)
    elif config.architecture == "gru":
        model = GRU_model_b1(seq_len=config.sequence_length)
    elif config.architecture == "simple_cnn":
        model = Simple_CNN(seq_len=config.sequence_length)
    else:
        raise ValueError("Unknown model type: %s" % config.architecture)
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
    print(summary(model, input_size=(config.BATCH_SIZE, seq_len, 256), verbose=0))
    # Loss functions
    criterions = [RMSELoss(),
                  RMSELoss()]
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,
                                                             eta_min=config.LR_MIN_CYCLIC),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=5,
                                                                                                 eta_min=config.LR_MIN_CYCLIC),
                                         len_loader=len(train_generator),
                                         warmup_steps=config.WARMUP_STEPS,
                                         warmup_start_lr=config.LR_MIN_WARMUP,
                                         warmup_mode=config.WARMUP_MODE)
    }
    # if we use discriminative learning, we don't need LR scheduler
    lr_scheduller = lr_schedullers[config.LR_SCHEDULLER]
    # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
    if config.LR_SCHEDULLER == 'Warmup_cyclic':
        optimizer.param_groups[0]['lr'] = config.LR_MIN_WARMUP

    # evaluation aprams
    best_val_RMSE = np.inf
    evaluation_metrics = {'MSE_val_arousal': mean_squared_error,
                            'MAE_val_arousal': mean_absolute_error,
                            'MSE_val_valence': mean_squared_error,
                          'MAE_val_valence': mean_absolute_error,
                          }
    # early stopping
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="min")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        print('------------------------')
        train_loss = train_epoch(model, train_generator, optimizer, criterions, device, print_step=50,
                                 warmup_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics = evaluate_model(model, dev_generator, evaluation_metrics, device)
        val_RMSE_arousal = val_metrics['MSE_val_arousal'] ** 0.5
        val_RMSE_valence = val_metrics['MSE_val_valence'] ** 0.5
        val_RMSE = (val_RMSE_arousal + val_RMSE_valence) / 2

        # update best val metrics got on validation set and log them using wandb
        if val_RMSE < best_val_RMSE:
            best_val_RMSE = val_RMSE
            wandb.config.update({'best_val_RMSE': best_val_RMSE}, allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model.pth'))
        print("Development RMSE (Arousal + Valence):%f" % val_RMSE)

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log({'val_MSE_arousal': val_metrics['MSE_val_arousal']}, commit=False)
        wandb.log({'val_MAE_arousal': val_metrics['MAE_val_arousal']}, commit=False)
        wandb.log({'val_MSE_valence': val_metrics['MSE_val_valence']}, commit=False)
        wandb.log({'val_MAE_valence': val_metrics['MAE_val_valence']}, commit=False)
        wandb.log({'val_RMSE_arousal': val_RMSE_arousal}, commit=False)
        wandb.log({'val_RMSE_valence': val_RMSE_valence}, commit=False)
        wandb.log({'val_RMSE': val_RMSE}, commit=False)
        wandb.log({'train_loss': train_loss})
        # update LR if needed
        if config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(val_RMSE)
        elif config.LR_SCHEDULLER == 'Cyclic':
            lr_scheduller.step()
        # check early stopping
        early_stopping_result = early_stopping_callback(val_RMSE, model)
        if early_stopping_result:
            print("Early stopping")
            break
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(window_size, stride, base_model_type, seq2one_model_type, batch_size):
    print("Start of the script....")
    # get data loaders
    train_generator, dev_generator, test_generator = get_data_loaders(window_size=int(window_size),
                                                                                       stride=int(stride),
                                                                                       base_model_type=base_model_type,
                                                                                       batch_size=batch_size)
    # train the model
    train_model(train_generator=train_generator, dev_generator=dev_generator,
                base_model_type=base_model_type, seq2one_model_type = seq2one_model_type,
                BATCH_SIZE=batch_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Emotion Recognition model training',
        epilog='Parameters: model_type, batch_size')
    parser.add_argument('--window_size', type=float, required=True)
    parser.add_argument('--stride', type=float, required=True)
    parser.add_argument('--base_model_type', type=str, required=True)
    parser.add_argument('--seq2one_model_type', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    # turn passed args from int to bool
    print("Passed args: ", args)
    # check arguments
    if args.base_model_type not in ['EfficientNet-B1', 'EfficientNet-B4']:
        raise ValueError("model_type should be either EfficientNet-B1 or EfficientNet-B4. Got %s" % args.base_model_type)
    if args.batch_size < 1:
        raise ValueError("batch_size should be greater than 0")
    # convert args to bool
    window_size = args.window_size
    stride = args.stride
    base_model_type = args.base_model_type
    seq2one_model_type = args.seq2one_model_type
    batch_size = args.batch_size
    # run main script with passed args
    main(window_size=window_size, stride=stride,
         base_model_type = base_model_type, seq2one_model_type = seq2one_model_type,
         batch_size=batch_size)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()

