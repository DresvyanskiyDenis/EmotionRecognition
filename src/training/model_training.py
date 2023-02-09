import gc
import os
from functools import partial
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, mean_squared_error, \
    mean_absolute_error
from torch.nn.functional import one_hot
from torchinfo import summary

import training_config
from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.models.CNN_models import Modified_MobileNetV3_large
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping
from pytorch_utils.training_utils.losses import SoftFocalLoss
from pytorch_utils.models.ViT_models import ViT_Deit_model
from src.training.data_preparation import load_data_and_construct_dataloaders, calculate_class_weights


def evaluate_model(model:torch.nn.Module, generator:torch.utils.data.DataLoader, device:torch.device) -> Tuple[Dict[object, float],...]:

    evaluation_metrics_classification = { 'accuracy': accuracy_score,
                                          'precision': partial(precision_score, average='macro'),
                                          'recall': partial(recall_score, average='macro'),
                                          'f1': partial(f1_score, average='macro')
                                        }

    evaluation_metric_arousal = {'mse': mean_squared_error,
                                 'mae': mean_absolute_error
                                 }

    evaluation_metric_valence = {'mse': mean_squared_error,
                                 'mae': mean_absolute_error
                                 }
    # create arrays for predictions and ground truth labels
    predictions_classifier, predictions_arousal, predictions_valence = [],[],[]
    ground_truth_classifier, ground_truth_arousal, ground_truth_valence = [],[],[]

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # separate labels into a list of torch.Tensor
            labels = [labels[:,0], labels[:,1], labels[:,2]]
            labels[2] = one_hot(labels[2].long(), num_classes=training_config.NUM_CLASSES)
            labels = [label.float().to(device) for label in labels]

            # forward pass
            output_classifier, output_arousal, output_valence = model(inputs)
            output_classifier = torch.softmax(output_classifier, dim=-1)
            output_classifier = torch.argmax(output_classifier, dim=-1)
            output_classifier = output_classifier.cpu().numpy().squeeze()
            output_arousal = output_arousal.cpu().numpy().squeeze()
            output_valence = output_valence.cpu().numpy().squeeze()
            # save ground_truth labels and predictions in arrays
            predictions_arousal.append(output_arousal)
            predictions_valence.append(output_valence)
            predictions_classifier.append(output_classifier)
            ground_truth_arousal.append(labels[0].cpu().numpy().squeeze())
            ground_truth_valence.append(labels[1].cpu().numpy().squeeze())
            ground_truth_classifier.append(torch.argmax(labels[2], dim=-1).cpu().numpy().squeeze())
        # concatenate evaluation metrics
        predictions_arousal = np.concatenate(predictions_arousal)
        predictions_valence = np.concatenate(predictions_valence)
        predictions_classifier = np.concatenate(predictions_classifier)
        ground_truth_arousal = np.concatenate(ground_truth_arousal)
        ground_truth_valence = np.concatenate(ground_truth_valence)
        ground_truth_classifier = np.concatenate(ground_truth_classifier)
        # calculate evaluation metrics
        evaluation_metrics_arousal = {metric: evaluation_metric_arousal[metric](ground_truth_arousal, predictions_arousal) for metric in evaluation_metric_arousal}
        evaluation_metrics_valence = {metric: evaluation_metric_valence[metric](ground_truth_valence, predictions_valence) for metric in evaluation_metric_valence}
        evaluation_metrics_classifier = {metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier) for metric in evaluation_metrics_classification}
        # print evaluation metrics
        print('Evaluation metrics for arousal:')
        for metric_name, metric_value in evaluation_metrics_arousal.items():
            print("%s: %.4f" % (metric_name, metric_value))
        print('Evaluation metrics for valence:')
        for metric_name, metric_value in evaluation_metrics_valence.items():
            print("%s: %.4f" % (metric_name, metric_value))
        print('Evaluation metrics for classifier:')
        for metric_name, metric_value in evaluation_metrics_classifier.items():
            print("%s: %.4f" % (metric_name, metric_value))
    return (evaluation_metrics_arousal, evaluation_metrics_valence, evaluation_metrics_classifier)



def train_step(model:torch.nn.Module, criterion:Tuple[torch.nn.Module,...],
               inputs:Tuple[torch.Tensor,...], ground_truths:List[torch.Tensor]) -> List:
    """ Performs one training step for a model.

    :param model: torch.nn.Module
            Model to train.
    :param criterion: Tuple[torch.nn.Module,...]
            Loss functions for each output of the model.
    :param inputs: Tuple[torch.Tensor,...]
            Inputs for the model.
    :param ground_truths: Tuple[torch.Tensor,...]
            Ground truths for the model. Should be in the same order as the outputs of the model.
            Some elements can be NaN if the corresponding output is not used for training (label does not exist).
    :return:
    """
    # forward pass
    outputs = model(*inputs)
    losses = []
    # TODO: check if masking works right
    for i, criterion_i in enumerate(criterion):
        # calculate mask for current loss function
        mask = ~torch.isnan(ground_truths[i])
        # calculate loss based on mask
        loss = criterion_i(outputs[i][mask], ground_truths[i][mask])
        losses.append(loss)

    return losses


def train_epoch(model:torch.nn.Module, train_generator:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer, criterions:Tuple[torch.nn.Module,...],
               device:torch.device, print_step:int=100,
               accumulate_gradients:Optional[int]=1,
               warmup_lr_scheduller:Optional[torch.optim.lr_scheduler]=None)->float:
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

        # separate labels into a list of torch.Tensor
        labels = [labels[:,0], labels[:,1], labels[:,2]]
        labels[2] = one_hot(labels[2].long(), num_classes=training_config.NUM_CLASSES)
        labels = [label.float().to(device) for label in labels]

        # do train step
        with torch.set_grad_enabled(True):
            step_losses = train_step(model, criterions, inputs, labels)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # backward pass
            for step_loss in step_losses:
                step_loss.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_lr_scheduller is not None:
                    warmup_lr_scheduller.step()

        # print statistics
        running_loss += sum([loss.item() for loss in step_losses])
        total_loss += sum([loss.item() for loss in step_losses])
        counter += 1
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
    return total_loss / counter


def train_model(train_generator:torch.utils.data.DataLoader, dev_generator:torch.utils.data.DataLoader,
                class_weights:torch.Tensor) -> None:

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Modified_MobileNetV3_large(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                       num_regression_neurons=training_config.NUM_REGRESSION_NEURONS)
    model = model.to(device)
    summary(model, input_size=(training_config.BATCH_SIZE, 3, training_config.IMAGE_RESOLUTION[0], training_config.IMAGE_RESOLUTION[1]))
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[training_config.OPTIMIZER](model.parameters(), lr=training_config.LR_MAX_CYCLIC, weight_decay=training_config.WEIGHT_DECAY)
    # Loss functions
    criterions = (torch.nn.MSELoss(), torch.nn.MSELoss(), SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2))
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.ANNEALING_PERIOD,
                                                             eta_min=training_config.LR_MIN_CYCLIC),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic' : WarmUpScheduler(optimizer=optimizer,
                                          lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=training_config.ANNEALING_PERIOD,
                                                             eta_min=training_config.LR_MIN_CYCLIC),
                                         len_loader = len(train_generator),
                                          warmup_steps=training_config.WARMUP_STEPS,
                                          warmup_start_lr = training_config.LR_MIN_WARMUP,
                                          warmup_mode = training_config.WARMUP_MODE)
    }
    lr_scheduller = lr_schedullers[training_config.LR_SCHEDULLER]
    # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
    optimizer.param_groups[0]['lr'] = training_config.LR_MIN_WARMUP
    # callbacks
    val_metrics = {
        'val_recall': partial(recall_score, average='macro'),
        'val_precision': partial(precision_score, average='macro'),
        'val_f1_score': partial(f1_score, average='macro')
    }
    best_val_metric_value = np.inf # we do minimization
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=training_config.EARLY_STOPPING_PATIENCE,
                                                 save_path='best_model/',
                                                 mode="min")

    # train model
    for epoch in range(training_config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterions, device, print_step=100)
        print("Train loss: %.10f" % train_loss)
        # validate
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metric_arousal, val_metric_valence, val_metrics_classification = evaluate_model(model, dev_generator, device)
        metric_value = val_metric_arousal['mse']+val_metric_valence['mse']+(1-val_metrics_classification['recall'])
        # update LR
        if training_config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(metric_value)
        else:
            lr_scheduller.step()
        # save best model
        if metric_value < best_val_metric_value:
            best_val_metric_value = metric_value
            torch.save(model.state_dict(), os.path.join(training_config.BEST_MODEL_SAVE_PATH, 'best_model.pth'))
        # check early stopping
        early_stopping_result = early_stopping_callback(metric_value, model)
        if early_stopping_result:
            print("Early stopping")
            break
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()



def main():
    # get data loaders
    train_generator, dev_generator, class_weights = load_data_and_construct_dataloaders()
    # calculate class_weights
    class_weights = calculate_class_weights(train_generator)
    # train model
    train_model(train_generator=train_generator, dev_generator=dev_generator,
    class_weights=class_weights)



if __name__ == "__main__":
    main()

