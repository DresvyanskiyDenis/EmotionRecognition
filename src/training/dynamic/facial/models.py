import torch
from fastai.layers import TimeDistributed

from pytorch_utils.layers.attention_layers import Transformer_layer
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1


class Transformer_model_b1(torch.nn.Module):
    """

    """
    def __init__(self, path_to_weights_base_model:str):
        super(Transformer_model_b1, self).__init__()
        self.base_model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8,
                                              num_regression_neurons=2)
        self.base_model.load_state_dict(torch.load(path_to_weights_base_model))
        # cut off last two layers of base model
        self.base_model = torch.nn.Sequential(*list(self.base_model.children())[:-2])
        # freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        # wrap model in TimeDistributed layer
        self.base_model = TimeDistributed(self.base_model)
        # create three transformer blocks
        self.transformer_block_1 = Transformer_layer(input_dim=256, num_heads=8,
                                                     positional_encoding=True)
        self.transformer_block_2 = Transformer_layer(input_dim=256, num_heads=8,
                                                        positional_encoding=True)
        # create linear layer
        self.linear = torch.nn.Linear(256, 2)
        # create tanh activation layer
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # base model
        x = self.base_model(x)
        # transformer blocks
        x = self.transformer_block_1(x, x, x)
        x = self.transformer_block_2(x, x, x)
        # linear layer
        x = self.linear(x)
        # tanh activation
        x = self.tanh(x)
        return x


class GRU_model_b1(torch.nn.Module):
    def __init__(self, path_to_weights_base_model:str):
        super(GRU_model_b1, self).__init__()
        self.base_model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8,
                                              num_regression_neurons=2)
        self.base_model.load_state_dict(torch.load(path_to_weights_base_model))
        # cut off last two layers of base model
        self.base_model = torch.nn.Sequential(*list(self.base_model.children())[:-2])
        # freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        # wrap model in TimeDistributed layer
        self.base_model = TimeDistributed(self.base_model)
        # create GRU layer
        self.gru = torch.nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
        # create linear layer
        self.linear = torch.nn.Linear(128, 2)
        # create tanh activation layer
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # base model
        x = self.base_model(x)
        # GRU
        x, _ = self.gru(x)
        # linear layer
        x = self.linear(x)
        # tanh activation
        x = self.tanh(x)
        return x

    def print_weights(self):
        print(self.linear.weight)