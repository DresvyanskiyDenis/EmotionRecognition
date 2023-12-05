import torch
from fastai.layers import TimeDistributed
from torchinfo import summary

from pytorch_utils.layers.attention_layers import Transformer_layer
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1


class Transformer_model_b1(torch.nn.Module):
    """

    """
    def __init__(self, seq_len:int):
        super(Transformer_model_b1, self).__init__()
        self.seq_len = seq_len
        # create three transformer blocks
        self.transformer_block_1 = Transformer_layer(input_dim=256, num_heads=8,
                                                     positional_encoding=True)
        self.transformer_block_2 = Transformer_layer(input_dim=256, num_heads=8,
                                                        positional_encoding=True)
        # get rid of sequence dimension using conv1d
        self.conv1d = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=self.seq_len)
        # batch norm
        self.batch_norm = torch.nn.BatchNorm1d(num_features=128)
        # create linear layer
        self.linear = torch.nn.Linear(128, 2)
        # create tanh activation layer
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # transformer blocks
        x = self.transformer_block_1(x, x, x)
        x = self.transformer_block_2(x, x, x) # output shape: (batch_size, seq_len, hidden_size)
        # permute x to (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)
        # conv1d
        x = self.conv1d(x)
        # squeeze sequence dimension
        x = x.squeeze(dim=2)
        # batch norm
        x = self.batch_norm(x)
        # linear layer
        x = self.linear(x)
        # tanh activation
        x = self.tanh(x)
        return x


class GRU_model_b1(torch.nn.Module):
    def __init__(self, seq_len:int):
        super(GRU_model_b1, self).__init__()
        self.seq_len = seq_len
        # create GRU layer
        self.gru = torch.nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
        # get rid of sequence dimension using conv1d
        self.conv1d = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.seq_len)
        # batch norm
        self.batch_norm = torch.nn.BatchNorm1d(num_features=128)
        # create linear layer
        self.linear = torch.nn.Linear(128, 2)
        # create tanh activation layer
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # GRU
        x, _ = self.gru(x) # output: (batch_size, seq_len, hidden_size)
        # change the shape of x to (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)
        # conv1d
        x = self.conv1d(x)
        # squeeze sequence dimension
        x = x.squeeze(dim=-1)
        # batch norm
        x = self.batch_norm(x)
        # linear layer
        x = self.linear(x)
        # tanh activation
        x = self.tanh(x)
        return x

    def print_weights(self):
        print(self.linear.weight)


class Simple_CNN(torch.nn.Module):
    def __init__(self, seq_len:int):
        super(Simple_CNN, self).__init__()
        self.seq_len = seq_len
        # create conv1d layer
        self.conv1d = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=self.seq_len)
        # batch norm
        self.batch_norm = torch.nn.BatchNorm1d(num_features=128)
        # create linear layer
        self.linear = torch.nn.Linear(128, 2)
        # create tanh activation layer
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # permute x to (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)
        # conv1d
        x = self.conv1d(x)
        # squeeze sequence dimension
        x = x.squeeze(dim=-1)
        # batch norm
        x = self.batch_norm(x)
        # linear layer
        x = self.linear(x)
        # tanh activation
        x = self.tanh(x)
        return x



if __name__ == "__main__":
    model = Simple_CNN(seq_len=15)
    summary(model, input_size=(2, 15, 256))

    model = GRU_model_b1(seq_len=15)
    summary(model, input_size=(2, 15, 256))

    model = Transformer_model_b1(seq_len=15)
    summary(model, input_size=(2, 15, 256))