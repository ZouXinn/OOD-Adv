'''
This file realizes the universal adversarial attacks
'''
import time
from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class UAP(nn.Module):
    def __init__(self,
                shape=(28, 28),
                num_channels=1):
        super(UAP, self).__init__()
        
        self.num_channels = num_channels
        self.shape = shape
        
        # Controls when normalization is used.
        self.normalization_used = None
        self._normalization_applied = None
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))
            
            
    def set_normalization_used(self, mean, std):
        self.normalization_used = {}
        n_channels = len(mean)
        assert n_channels == self.num_channels, f"The channel number of the params is not consistent with that of the UAP, which is {self.num_channels}"
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used['mean'] = mean
        self.normalization_used['std'] = std
        self._normalization_applied = True
        
    def normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs*std + mean

    def forward(self, x):
        uap = self.uap
        # Put image into original form
        if self._normalization_applied:
            orig_img = self.inverse_normalize(x)
            # Add uap to input
            adv_orig_img = orig_img + uap
            # Put image into normalized form
            adv_x = self.normalize(adv_orig_img)
        else:
            adv_x = x + uap
        return adv_x



class UniversalAttack(object):
    r"""
    Base class for all universal attacks.
    .. note::
        It automatically set device to the device where given model is.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name

        self.model = model
        self.device = next(model.parameters()).device # set the device automatically
    
        

    
    def train(self, uap, dataloader):
        r"""
        @function: It attacks the model with datas in dataloader, i.e. trains the UAP
        @parameters:
            @uap: The uap to attack and return
            @dataloader: The data loader of datas used to conduct attack
        @return: The uap
        """
        raise NotImplementedError
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count