


import warnings

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
import torch.nn as nn
from ctgain.data_sampler_v2 import DataSampler


from ctgain.utils.ctgain import *


from ctgain.ctgan.synthesizers.base import BaseSynthesizer, random_state



class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, nan_mask_dim,discriminator_dim, device, pac=1, h_dim=32,hint_rate=.8):
        super(Discriminator, self).__init__()
        self._device=device
        self.nan_mask_dim=nan_mask_dim

        dim = (input_dim+nan_mask_dim) #* pac #include the mask vector 
        self.pac = pac
        self.hint_rate=.7

        self.h_dim=h_dim

        self.uniform = torch.distributions.Uniform(low=0, high=1)
        # self.pacdim = dim

        seq = []
        # print(dim)

        seq+=[Linear(dim, dim), LeakyReLU(.2), Dropout(.5)]
        

        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        
        seq += [Linear(dim, dim), LeakyReLU(.2), Dropout(.5)]
        seq += [Linear(dim, dim), LeakyReLU(.2), Dropout(.5)]
        seq += [Linear(dim, dim), LeakyReLU(.2), Dropout(.5)]
        

        seq += [Linear(dim, nan_mask_dim),
                 nn.Sigmoid()]  # before was set to 74
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_, mask):
        """Apply the Discriminator to the `input_`."""
        # assert input_.size()[0] % self.pac == 0
        x_hat=input_
        
        hint = (self.uniform.sample([mask.shape[0], mask.shape[1]]) < self.hint_rate).float().to(self._device)
        hint = mask * hint ## the size of the hint matrix is wrong 
        
        
        inp = torch.cat([x_hat, hint], dim=1)
        
        # print( "#### 78 ####line ----->",inp.view(-1, self.pacdim).shape)
        
        return self.seq(inp)#.view(-1, self.pacdim))
        #return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_, pred=False):
        """Apply the Residual layer to the `input_`."""
        
        # print(1)
        if pred:
            print(input_.dtype)
            input_=input_.to(torch.float32)#changed from float64

        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        # print(" output ", out.shape, " input ",input_.shape)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim, nan_mask_dim, device):
        super(Generator, self).__init__()
        self._device=device
        self.seed_sampler = torch.distributions.Uniform(low=0, high=0.01)
        self.h_dim = embedding_dim
        # print("line 99 --- > embedding dims: ", embedding_dim)

        dim = data_dim # include the mask vector

        
        # seq = [Residual(dim, data_dim*2)]

        i=0
        # for item in list(generator_dim):
        #     seq += [Residual(dim , item)]
        #     print( dim , item)
        #     dim += item
        dim
        ## beginning trapianto 
        inpt_dimension=dim + nan_mask_dim
        inpt_dimension=dim*2

        generator = [Residual(inpt_dimension, dim)]

        
        # generator = [Linear(dim*2, dim*4),nn.Tanh()]

        # generator = [Residual(dim , dim)]
        
        generator.extend([Residual(inpt_dimension+dim, dim)])

        # generator.extend([Linear(dim*4, dim*4), nn.Tanh()])

        # generator.extend([Residual(dim*4, dim)])

        # generator.extend([Linear(dim*5, dim*2), nn.ReLU()])
        generator.extend([Linear( inpt_dimension+dim*2,  dim), nn.LeakyReLU(.2)])

        generator.extend([Linear( dim,  dim), nn.LeakyReLU(.2)])
        for layer in generator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.seq = Sequential(*generator)


        # ## end trapianto
        # seq+=[Residual(dim, data_dim)]

        # # i+=1
        
        # seq.append(Linear(dim, data_dim))

        # print( "line 117 eccolo ", (dim , data_dim))
        # print(i)
        # self.seq = Sequential(*seq)

    def forward(self, input_, mask, sample_nan_mask, pred=False):
        """Apply the Generator to the `input_`."""
        data_norm=input_

        data_norm[mask==0]=0
        
        # mask=torch.from_numpy(mask).to(self._device)
        # print("line 114 ---> data norm shape: ", data_norm.shape)
        z = self.seed_sampler.sample([data_norm.shape[0], data_norm.shape[1]]).to(self._device)
        # z = self.seed_sampler.sample([data_norm.shape[0], self.h_dim]).to(self._device)
        
        # print("line 116 ----> z shape: ", z.shape)
        # print( type(mask), type(z))
        
        # mask=torch.from_numpy(mask).to(self._device)
        # if mask.shape[0] != z.shape[0]:
        #     mask=mask.repeat(z.shape[0], 1)
        #     print( mask.shape, z.shape)
        # if type(mask)==np.ndarray:
        #     mask=torch.from_numpy(mask).to(self._device)                     
        z=z.to(self._device)
        mask=mask.to(self._device)
        data_norm.to(self._device)
        random_combined = mask * data_norm + (1-mask) * z
        # if type(random_combined)==np.ndarray:
        #     print( type(random_combined), type(mask))
        #     random_combined=torch.from_numpy(random_combined).to(self._device)
        #     mask=torch.from_numpy(mask).to(self._device)
        # if random_combined.dtype!= mask.dtype:
        #     random_combined=random_combined.type(mask.dtype)
        
        xoxo=torch.cat([random_combined, mask], dim=1)
        
        
        
        sample = self.seq(xoxo)
        # x_hat = random_combined * (mask) + sample * (1-mask)
        # data = self.seq(input_)

        return sample, random_combined



