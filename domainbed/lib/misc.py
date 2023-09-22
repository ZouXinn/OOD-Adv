# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
from collections import Counter
from itertools import cycle


def distance(h1, h2):
    ''' distance of two networks (h1, h2 are classifiers)'''
    dist = 0.
    for param in h1.state_dict():
        h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
        dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
    return torch.sqrt(dist)

def proj(delta, adv_h, h):
    ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
    ''' adv_h and h are two classifiers'''
    dist = distance(adv_h, h)
    if dist <= delta:
        return adv_h
    else:
        ratio = delta / dist
        for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
            param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
        # print("distance: ", distance(adv_h, h))
        return adv_h

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches): # get random (constrained) pairs of domains
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0 # recurrent shift, j = (i + 1) % len(minibatchs)

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1] # xi is the batch_x in env perm[i], and yi is the batch_y in env perm[i]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
        # xi, xj comes from different environments

        min_n = min(len(xi), len(xj)) # minimal batch size

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist() # permutate the domains randomly
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)] # the domains used for meta train
    meta_test = perm[-num_meta_test:] # the domains used for meta test

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1] # the meta training batch
         xj, yj = minibatches[j][0], minibatches[j][1] # the meta testing batch

         min_n = min(len(xi), len(xj)) # choose the smallest size
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device): # calculate the accuracy of network in loader
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    if hasattr(network, "eval_attacker"):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = network.eval_attacker(x, y)
            network.eval()
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1: # binary classification
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    else:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset : weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)
                if p.size(1) == 1: # binary classification
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
    network.train()

    return correct / total

def clean_accuracy(network, loader, weights, device): # calculate the accuracy of network in loader
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1: # binary classification
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def adversarial_accuracy(network, attack, loader, weights, device): # calculate the adversarial accuracy of network in loader
    correct = 0
    total = 0
    weights_offset = 0
    network.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x_adv = attack(x, y)
        p_adv = network.predict(x_adv)
        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if p_adv.size(1) == 1: # binary classification
            correct += (p_adv.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
        else:
            correct += (p_adv.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()
    network.train()
    
    return correct / total

def clean_adv_accuracy(network, attack, loader, weights, device):
    correct = 0
    correct_adv = 0
    total = 0
    weights_offset = 0
    network.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x_adv = attack(x, y)
        p = network.predict(x)
        p_adv = network.predict(x_adv)
        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if p_adv.size(1) == 1: # binary classification
            correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            correct_adv += (p_adv.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
        else:
            correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            correct_adv += (p_adv.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()
    network.train()
    
    return correct / total, correct_adv / total

def universal_adversarial_correct(network, uap, loader, weights, device):
    r'''
    @function: return the number of correctness of a dataloader
    @parameters:
        @network: the network to be attacked
        @uap: the uap to conduct the universal adversarial attack
        @loader: the datas to attack
        @weights: the weights on the data in the loader
        @device: the device to put the data on
    @return: correct, total
        @correct: the number of examples that are correctly classified under uap (weighted)
        @total: the total number of examples in the loader
    '''
    correct = 0
    total = 0
    weights_offset = 0
    network_training = network.training
    uap_training = uap.training
    network.eval()
    uap.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x_adv = uap(x)
        # clamp
        x_adv.data = torch.clamp(x_adv.data, 0, 1)
        p_adv = network.predict(x_adv)
        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if p_adv.size(1) == 1: # binary classification
            correct += (p_adv.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
        else:
            correct += (p_adv.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()
    
    if uap_training:
        uap.train()
    if network_training:
        network.train()
    return correct, total

def classwise_universal_adversarial_accuracy(network, uaps, loader, weights, device):
    r'''
    @function: return the adversarial accuracy of a dataloader under the classwise uap
    @parameters:
        @network: the network to be attacked
        @uaps: the list of uaps to conduct the universal adversarial attack, the length is class-number
        @loader: the datas to attack
        @weights: the weights on the data in the loader
        @device: the device to put the data on
    @return: correct, total
        @accuracy: the classwise universal adversarial accuracy
    '''
    correct = 0
    total = 0
    weights_offset = 0
    network_training = network.training
    uaps_training = [uap.training for uap in uaps]
    network.eval()
    for uap in uaps:
        uap.eval()
        
    num_classes = len(uaps)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        for cls in range(num_classes):
            mask_cls = (y==cls).nonzero().squeeze(dim=1)
            perturbed_x = uaps[cls](x[mask_cls])
            perturbed_x = torch.clamp(perturbed_x, 0, 1) # clamp the data to [0,1]
            x[mask_cls] = perturbed_x
        
        p_adv = network.predict(x)
        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if p_adv.size(1) == 1: # binary classification
            correct += (p_adv.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
        else:
            correct += (p_adv.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()
    
    for i, uap_training in enumerate(uaps_training):
        if uap_training:
            uaps[i].train()
    if network_training:
        network.train()
    return correct/total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
        
        
class MyTee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        if self.file:
            self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        if self.file:
            self.file.flush()
    
    def close_file(self):
        self.file.close()
        self.file = None

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
