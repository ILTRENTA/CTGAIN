import torch 
import torch.nn as nn

from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional



## a mix between CTGAN and GAIN 
class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class CTGAINGenerator(nn.Module):
    def __init__(self, data_dim,generator_dim, args, transform_params):

        super(CTGAINGenerator, self).__init__()
        dim= data_dim*2
        seq=[]

        for item in list(generator_dim):
            seq+=[Residual(dim, item)]
            dim+=item
            