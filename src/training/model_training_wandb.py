import argparse
import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/emotion_recognition_project/"])
import gc
import os
from functools import partial
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, mean_squared_error, \
    mean_absolute_error
from torch.nn.functional import one_hot

import training_config
from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.models.CNN_models import Modified_MobileNetV3_large
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping, GradualLayersUnfreezer, gradually_decrease_lr
from pytorch_utils.training_utils.losses import SoftFocalLoss, RMSELoss
from src.training.data_preparation import load_data_and_construct_dataloaders
import wandb


def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device) -> Tuple[
    Dict[object, float], ...]:
    evaluation_metrics_classification = {'val_accuracy_classification': accuracy_score,
                                         'val_precision_classification': partial(precision_score, average='macro'),
                                         'val_recall_classification': partial(recall_score, average='macro'),
                                         'val_f1_classification': partial(f1_score, average='macro')
                                         }

    evaluation_metric_arousal = {'val_arousal_rmse': mean_squared_error,
                                 'val_arousal_mae': mean_absolute_error
                                 }

    evaluation_metric_valence = {'val_valence_rmse': mean_squared_error,
                                 'val_valence_mae': mean_absolute_error
                                 }
    # create arrays for predictions and ground truth labels
    predictions_classifier, predictions_arousal, predictions_valence = [], [], []
    ground_truth_classifier, ground_truth_arousal, ground_truth_valence = [], [], []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)

            # forward pass
            outputs = model(inputs)
            regression_output = [outputs[1][:, 0], outputs[1][:, 1]]
            classification_output = outputs[0]

            # transform classification output to fit labels
            classification_output = torch.softmax(classification_output, dim=-1)
            classification_output = classification_output.cpu().numpy().squeeze()
            classification_output = np.argmax(classification_output, axis=-1)
            # transform regression output to fit labels
            regression_output = [regression_output[0].cpu().numpy().squeeze(),
                                 regression_output[1].cpu().numpy().squeeze()]

            # transform ground truth labels to fit predictions and sklearn metrics
            classification_ground_truth = labels[:, 2].cpu().numpy().squeeze()
            regression_ground_truth = [labels[:, 0].cpu().numpy().squeeze(), labels[:, 1].cpu().numpy().squeeze()]

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions_arousal.append(regression_output[0])
            predictions_valence.append(regression_output[1])
            predictions_classifier.append(classification_output)
            ground_truth_arousal.append(regression_ground_truth[0])
            ground_truth_valence.append(regression_ground_truth[1])
            ground_truth_classifier.append(classification_ground_truth)

        # concatenate all predictions and ground truth labels
        predictions_arousal = np.concatenate(predictions_arousal, axis=0)
        predictions_valence = np.concatenate(predictions_valence, axis=0)
        predictions_classifier = np.concatenate(predictions_classifier, axis=0)
        ground_truth_arousal = np.concatenate(ground_truth_arousal, axis=0)
        ground_truth_valence = np.concatenate(ground_truth_valence, axis=0)
        ground_truth_classifier = np.concatenate(ground_truth_classifier, axis=0)

        # create mask for all NaN values to remove them from evaluation
        mask_arousal = ~np.isnan(ground_truth_arousal)
        mask_valence = ~np.isnan(ground_truth_valence)
        mask_classifier = ~np.isnan(ground_truth_classifier)
        # remove NaN values from arrays
        predictions_arousal = predictions_arousal[mask_arousal]
        predictions_valence = predictions_valence[mask_valence]
        predictions_classifier = predictions_classifier[mask_classifier]
        ground_truth_arousal = ground_truth_arousal[mask_arousal]
        ground_truth_valence = ground_truth_valence[mask_valence]
        ground_truth_classifier = ground_truth_classifier[mask_classifier]

        # calculate evaluation metrics
        evaluation_metrics_arousal = {
            metric: evaluation_metric_arousal[metric](ground_truth_arousal, predictions_arousal) for metric in
            evaluation_metric_arousal}
        evaluation_metrics_valence = {
            metric: evaluation_metric_valence[metric](ground_truth_valence, predictions_valence) for metric in
            evaluation_metric_valence}
        evaluation_metrics_classifier = {
            metric: evaluation_metrics_classification[metric](ground_truth_classifier, predictions_classifier) for
            metric in evaluation_metrics_classification}
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


def train_step(model: torch.nn.Module, criterion: Tuple[torch.nn.Module, ...],
               inputs: Tuple[torch.Tensor, ...], ground_truths: List[torch.Tensor],
               device: torch.device) -> List:
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
    :param device: torch.device
            Device to use for training.
    :return:
    """
    # forward pass
    outputs = model(inputs)
    regression_output = [outputs[1][:, 0], outputs[1][:, 1]]
    classification_output = outputs[0]
    # criterions with indices 0 and 1 are for regression, 2 - for classification
    regression_criterion = criterion[0:2]
    classification_criterion = criterion[2]
    # prepare labels for loss calculation
    # separate labels into a list of torch.Tensor
    regression_labels = [ground_truths[:, 0], ground_truths[:, 1]]
    classification_labels = ground_truths[:, 2]
    # go through regression criterions
    regression_losses = []
    for i, criterion_i in enumerate(regression_criterion):
        # calculate mask for current loss function
        regression_mask = ~torch.isnan(regression_labels[i])
        output = regression_output[i][regression_mask]
        ground_truth = regression_labels[i][regression_mask]
        # calculate loss based on mask
        ground_truth = ground_truth.to(device)
        loss = criterion_i(output, ground_truth)
        regression_losses.append(loss)

    # go through classification criterion
    classification_mask = ~torch.isnan(classification_labels)
    output = classification_output[classification_mask]
    ground_truth = classification_labels[classification_mask]
    # turn into one-hot encoding
    ground_truth = one_hot(ground_truth.long(), num_classes=training_config.NUM_CLASSES)
    # calculate loss based on mask
    ground_truth = ground_truth.to(device)
    loss = classification_criterion(output, ground_truth)
    classification_losses = [loss]
    # combine losses into one array
    losses = [*regression_losses, *classification_losses]
    return losses


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterions: Tuple[torch.nn.Module, ...],
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

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            step_losses = train_step(model, criterions, inputs, labels, device)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # backward pass
            sum_losses = sum(step_losses)
            sum_losses.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_lr_scheduller is not None:
                    warmup_lr_scheduller.step()

        # print statistics
        running_loss += sum_losses.item()
        total_loss += sum_losses.item()
        counter += 1
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
    return total_loss / counter


def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                class_weights: torch.Tensor, GRADUAL_UNFREEZING:Optional[bool]=False,
                DISCRIMINATIVE_LEARNING:Optional[bool]=False) -> None:
    print("Start of the model training. Gradual_unfreezing:%s, Discriminative_lr:%s" % (GRADUAL_UNFREEZING,
                                                                                       DISCRIMINATIVE_LEARNING))
    # metaparams
    metaparams = {
        # general params
        "architecture": "MobileNetV3_large",
        "dataset": "RECOLA, SEWA, SEMAINE, AFEW-VA, AffectNet",
        "BEST_MODEL_SAVE_PATH": training_config.BEST_MODEL_SAVE_PATH,
        "NUM_WORKERS": training_config.NUM_WORKERS,
        # model architecture
        "NUM_CLASSES": training_config.NUM_CLASSES,
        "NUM_REGRESSION_NEURONS": training_config.NUM_REGRESSION_NEURONS,
        # training metaparams
        "NUM_EPOCHS": training_config.NUM_EPOCHS,
        "BATCH_SIZE": training_config.BATCH_SIZE,
        "OPTIMIZER": training_config.OPTIMIZER,
        "AUGMENT_PROB": training_config.AUGMENT_PROB,
        "EARLY_STOPPING_PATIENCE": training_config.EARLY_STOPPING_PATIENCE,
        "WEIGHT_DECAY": training_config.WEIGHT_DECAY,
        # LR scheduller params
        "LR_SCHEDULLER": training_config.LR_SCHEDULLER,
        "ANNEALING_PERIOD": training_config.ANNEALING_PERIOD,
        "LR_MAX_CYCLIC": training_config.LR_MAX_CYCLIC,
        "LR_MIN_CYCLIC": training_config.LR_MIN_CYCLIC,
        "LR_MIN_WARMUP": training_config.LR_MIN_WARMUP,
        "WARMUP_STEPS": training_config.WARMUP_STEPS,
        "WARMUP_MODE": training_config.WARMUP_MODE,
        # gradual unfreezing (if applied)
        "GRADUAL_UNFREEZING": GRADUAL_UNFREEZING,
        "UNFREEZING_LAYERS_PER_EPOCH": training_config.UNFREEZING_LAYERS_PER_EPOCH,
        "LAYERS_TO_UNFREEZE_BEFORE_START": training_config.LAYERS_TO_UNFREEZE_BEFORE_START,
        # discriminative learning
        "DISCRIMINATIVE_LEARNING": DISCRIMINATIVE_LEARNING,
        "DISCRIMINATIVE_LEARNING_INITIAL_LR": training_config.DISCRIMINATIVE_LEARNING_INITIAL_LR,
        "DISCRIMINATIVE_LEARNING_MINIMAL_LR": training_config.DISCRIMINATIVE_LEARNING_MINIMAL_LR,
        "DISCRIMINATIVE_LEARNING_MULTIPLICATOR": training_config.DISCRIMINATIVE_LEARNING_MULTIPLICATOR,
        "DISCRIMINATIVE_LEARNING_STEP": training_config.DISCRIMINATIVE_LEARNING_STEP,
        "DISCRIMINATIVE_LEARNING_START_LAYER": training_config.DISCRIMINATIVE_LEARNING_START_LAYER,
    }
    # initialization of Weights and Biases
    wandb.init(project="Emotion_recognition_F2F", config=metaparams)
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH':wandb.run.dir}, allow_val_change=True)

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Modified_MobileNetV3_large(embeddings_layer_neurons=256, num_classes=config.NUM_CLASSES,
                                       num_regression_neurons=config.NUM_REGRESSION_NEURONS)
    model = model.to(device)

    # define all model layers (params), which will be used by optimizer
    model_layers = [
        *list(list(list(model.children())[0].children())[0].children()),
        *list(list(model.children())[0].children())[1:],
        *list(model.children())[1:]
    ]
    # layers unfreezer
    if GRADUAL_UNFREEZING:
        layers_unfreezer = GradualLayersUnfreezer(model=model, layers=model_layers,
                                                  layers_per_epoch=config.UNFREEZING_LAYERS_PER_EPOCH,
                                                  layers_to_unfreeze_before_start=config.LAYERS_TO_UNFREEZE_BEFORE_START,
                                                  input_shape=(config.BATCH_SIZE, 3, training_config.IMAGE_RESOLUTION[0],
                                                               training_config.IMAGE_RESOLUTION[1]),
                                                  verbose=True)
    # if discriminative learning is applied
    if DISCRIMINATIVE_LEARNING:
        model_parameters = gradually_decrease_lr(layers=model_layers, initial_lr=config.DISCRIMINATIVE_LEARNING_INITIAL_LR,
                          multiplicator=config.DISCRIMINATIVE_LEARNING_MULTIPLICATOR, minimal_lr=config.DISCRIMINATIVE_LEARNING_MINIMAL_LR,
                          step=config.DISCRIMINATIVE_LEARNING_STEP, start_layer=config.DISCRIMINATIVE_LEARNING_START_LAYER)
        print('The learning rate was changed for each layer according to discriminative learning approach. The new learning rates are:')
    else:
        model_parameters = model.parameters()
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[config.OPTIMIZER](model_parameters, lr=config.LR_MAX_CYCLIC,
                                             weight_decay=config.WEIGHT_DECAY)
    # Loss functions
    class_weights = class_weights.to(device)
    criterions = (RMSELoss(), RMSELoss(), SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2))
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ANNEALING_PERIOD,
                                                             eta_min=config.LR_MIN_CYCLIC),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=config.ANNEALING_PERIOD,
                                                                                                 eta_min=config.LR_MIN_CYCLIC),
                                         len_loader=len(train_generator)//training_config.ACCUMULATE_GRADIENTS,
                                         warmup_steps=config.WARMUP_STEPS,
                                         warmup_start_lr=config.LR_MIN_WARMUP,
                                         warmup_mode=config.WARMUP_MODE)
    }
    # if we use discriminative learning, we don't need LR scheduler
    if DISCRIMINATIVE_LEARNING:
        lr_scheduller = None
    else:
        lr_scheduller = lr_schedullers[config.LR_SCHEDULLER]
        # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
        if config.LR_SCHEDULLER == 'Warmup_cyclic':
            optimizer.param_groups[0]['lr'] = config.LR_MIN_WARMUP

    # early stopping
    best_val_metric_value = -np.inf  # we do maximization
    best_val_rmse_arousal = np.inf
    best_val_rmse_valence = np.inf
    best_val_recall_classification = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="max")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterions, device, print_step=100,
                                 accumulate_gradients=training_config.ACCUMULATE_GRADIENTS,
                                 warmup_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metric_arousal, val_metric_valence, val_metrics_classification = evaluate_model(model, dev_generator,
                                                                                            device)

        # update best val metrics got on validation set and log them using wandb
        if val_metric_arousal['val_arousal_rmse'] < best_val_rmse_arousal:
            best_val_rmse_arousal = val_metric_arousal['val_arousal_rmse']
            wandb.config.update({'best_val_rmse_arousal': best_val_rmse_arousal}, allow_val_change=True)
        if val_metric_valence['val_valence_rmse'] < best_val_rmse_valence:
            best_val_rmse_valence = val_metric_valence['val_valence_rmse']
            wandb.config.update({'best_val_rmse_valence': best_val_rmse_valence}, allow_val_change=True)
        if val_metrics_classification['val_recall_classification'] > best_val_recall_classification:
            best_val_recall_classification = val_metrics_classification['val_recall_classification']
            wandb.config.update({'best_val_recall_classification': best_val_recall_classification}, allow_val_change=True)

        # calculate general metric as average of (1.-RMSE_arousal), (1.-RMSE_valence), and RECALL_classification
        metric_value = (1. - val_metric_arousal['val_arousal_rmse']) + \
                       (1. - val_metric_valence['val_valence_rmse']) + \
                       val_metrics_classification['val_recall_classification']
        metric_value = metric_value / 3.
        print("Validation metric (Average sum of (1.-RMSE_arousal), (1.-RMSE_valence), and RECALL_classification): %.10f" % metric_value)

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(val_metric_arousal, commit=False)
        wandb.log(val_metric_valence, commit=False)
        wandb.log(val_metrics_classification, commit=False)
        wandb.log({'val_general_metric': metric_value}, commit=False)
        wandb.log({'train_loss (rmse+rmse+crossentropy)': train_loss})
        # update LR if needed
        if config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(metric_value)
        elif config.LR_SCHEDULLER == 'Cyclic':
            lr_scheduller.step()

        # save best model
        if metric_value > best_val_metric_value:
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            best_val_metric_value = metric_value
            wandb.config.update({'best_val_metric_value\n(Average sum of (1.-RMSE_arousal), (1.-RMSE_valence), and RECALL_classification)':
                                     best_val_metric_value}, allow_val_change=True)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_metric.pth'))
        # check early stopping
        early_stopping_result = early_stopping_callback(metric_value, model)
        if early_stopping_result:
            print("Early stopping")
            break
        # unfreeze next n layers
        if GRADUAL_UNFREEZING:
            layers_unfreezer()
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(gradual_unfreezing, discriminative_learning):
    print("Start of the script....")
    # get data loaders
    (train_generator, dev_generator, test_generator), class_weights = load_data_and_construct_dataloaders(
        return_class_weights=True)
    # train the model
    train_model(train_generator=train_generator, dev_generator=dev_generator,class_weights=class_weights,
                GRADUAL_UNFREEZING=gradual_unfreezing, DISCRIMINATIVE_LEARNING=discriminative_learning)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Emotion Recognition model training',
        epilog='Two arguments are required: gradual_unfreezing and discriminative_learning. THey both have boolean type.')
    parser.add_argument('--gradual_unfreezing', type=int, required=True)
    parser.add_argument('--discriminative_learning', type=int, required=True)
    args = parser.parse_args()
    # turn passed args from int to bool
    print("Passed args: ",args.gradual_unfreezing, args.discriminative_learning)
    if args.gradual_unfreezing not in [0,1]:
        raise ValueError("gradual_unfreezing should be either 0 or 1")
    if args.discriminative_learning not in [0,1]:
        raise ValueError("discriminative_learning should be either 0 or 1")
    gradual_unfreezing = True if args.gradual_unfreezing == 1 else False
    discriminative_learning = True if args.discriminative_learning == 1 else False
    # run main script with passed args
    main(gradual_unfreezing, discriminative_learning)

