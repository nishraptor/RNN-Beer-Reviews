import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable

class baselineLSTM(nn.Module):
    def __init__(self, config):
        super(baselineLSTM, self).__init__()

        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.

        self.hidden_dim = config['hidden_dim']
        self.input_dim = config['input_dim']
        self.batch_size = config['batch_size']
        self.layers = config['layers']
        self.hidden = None

        if config['cuda']:
            computing_device = torch.device("cuda")
        else:
            computing_device = torch.device("cpu")

        self.init_hidden(computing_device)

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)





    def init_hidden(self, computing_device):

        self.hidden = None
        self.hidden = (torch.zeros(self.layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.layers, self.batch_size, self.hidden_dim))

        self.hidden = [tensor.to(computing_device) for tensor in self.hidden]

    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        out, self.hidden = self.lstm(sequence, self.hidden)

        return out