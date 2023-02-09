import torch


def calculate_class_weights(train_dataloader:torch.utils.data.DataLoader)->torch.Tensor:
    """ Calculate class weights from the data presented by the train_dataloader.

    Args:
        train_dataloader: torch.utils.data.DataLoader
                The dataloader for the training set.

    Returns: torch.Tensor
        The class weights calculated from the provided data. THe weights are calculated as
        1 / (number of samples in class / total number of samples). (Inverse proportion)

    """
    num_classes = next(iter(train_dataloader))[1].shape[1]
    class_weights = torch.zeros(num_classes)
    for _, labels in train_dataloader:
        class_weights += torch.sum(labels, dim=0)
    class_weights = class_weights / torch.sum(class_weights)
    class_weights = 1. / class_weights

    return class_weights
