# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc
from domainbed import datasets


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed, hp_dict=None):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    These are hyper-parameters for the algorithms and the networks, it is different from the args. They are parameters that affect the performance of the trained model
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    elif algorithm == 'RDANN':
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
    elif algorithm in ['AT', 'UAT', 'MAT', 'LDAT']: # this does not support RotatedMNIST, and we do not support NICO
        if dataset in ["RotatedMNIST", "NICO"]:
            raise NotImplementedError
        if algorithm in ['MAT', 'LDAT']:
            if dataset in SMALL_IMAGES:
                _hparam('is_cmnist', 1, lambda r: 1)
            else:
                _hparam('is_cmnist', 0, lambda r: 0)
        if dataset == "ColoredMNIST":
            _hparam('at_eps', 1, lambda r: 10**r.uniform(-1, -2))
            if algorithm == "AT":
                _hparam('at_alpha', 0.1, lambda r: 0.1)
            else: # MAT, LDAT, what about UAT?
                _hparam('at_alpha', 0.1, lambda r: 10**r.uniform(-2, 1))
        else:
            _hparam('at_eps', 0.1, lambda r: 0.1)
            _hparam('at_alpha', 0.1, lambda r: 0.1)
        _hparam('at_name', 0, lambda r: r.randint(10000))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL" or algorithm == "CausIRL_CORAL" or algorithm == "CausIRL_MMD":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm in ["MLDG"]:
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('n_meta_test', 2, lambda r:  int(r.choice([1, 2])))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10**r.uniform(-3, 5))

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10**r.uniform(1., 4.))
        _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "IB_ERM":
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "IB_IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "CAD" or algorithm == "CondCAD":
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)
        
    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    if algorithm in ["LDAT"]:
        if dataset == "ColoredMNIST":
            _hparam('at_cb_rank', 10, lambda r: int(r.uniform(10, 20)))#!codebook rank
            _hparam('A_lr', 0.01, lambda r: 0.01)
            _hparam('B_lr', 0.01, lambda r: 0.01)
        elif dataset == "NICO" or dataset == "RotatedMNIST":
            raise NotImplementedError
        else:
            _hparam('at_cb_rank', 8, lambda r: int(r.choice([5, 10, 15, 20])))#!codebook rank
            _hparam('A_lr', 0.1, lambda r: r.choice([0.1, 0.01]))
            _hparam('B_lr', 0.1, lambda r: r.choice([0.1, 0.01]))
    elif algorithm in ["MAT"]:
        if dataset == "ColoredMNIST":
            _hparam('at_delta_num', 8 , lambda r: int(r.uniform(5, 20)))
            _hparam('kat_alpha_step', -3,lambda r: 10**r.uniform(-3, -2))
        elif dataset == "NICO" or dataset == "RotatedMNIST":
            raise NotImplementedError
        else:
            _hparam('at_delta_num', 10 , lambda r: int(r.choice([5, 10, 15, 20])))
            _hparam('kat_alpha_step', 0.01, lambda r: r.choice([0.01, 0.001]))
        _hparam('KAT_num_iter', 1 , lambda r: 1)
        
    if algorithm in ["AERM", "RDANN"]:
        atk_hparam = _get_PGD_hparams(dataset, True)
        _hparam('atk_hparam', atk_hparam , lambda r: atk_hparam)
    

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))
            

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))

    if algorithm in ['DANN', 'CDANN', 'RDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN', 'RDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN', 'RDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN', 'RDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN', 'RDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN', 'RDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed): # the hyper-parameters are just depnedent on the algorithm, dataset, and seed
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}



def attack_hparams(attack, dataset, train=False):
    '''
    The hyper-parameters for adversarial attacks, which just depends on the dataset
    '''
    assert f'_get_{attack}_hparams' in globals(), f'Attack method {attack} not supported'
    assert dataset in _dataset_eps.keys(), f"The eps of dataset {dataset} is not set"
    return globals()[f'_get_{attack}_hparams'](dataset, train)

def uat_hparams(AT_method, dataset):
    '''
    The hyper-parameters for adversarial attacks, which just depends on the dataset
    '''
    assert f'_get_uat_{AT_method}_hparams' in globals(), f'UAT method {AT_method} not supported'
    assert dataset in _dataset_eps.keys(), f"The eps of dataset {dataset} is not set"
    return globals()[f'_get_uat_{AT_method}_hparams'](dataset)

def AT_batch_size(dataset, algorithm):
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']
    if dataset in SMALL_IMAGES:
        batch_size = 128
    elif algorithm == 'ARM':
        batch_size = 8
    elif dataset == 'DomainNet':
        batch_size = 32
    else:
        batch_size = 32
    return batch_size

_dataset_eps = {
    "RotatedMNIST": 0.1,
    "ColoredMNIST": 0.1,
    "PACS": 4/255,
    "VLCS": 4/255,
    "OfficeHome": 4/255,
    "TerraIncognita": 4/255,
    "DomainNet": 4/255,
}

_dateset_classes = {
    "RotatedMNIST": 10,
    "ColoredMNIST": 2,
    "PACS": 7,
    "VLCS": 5,
    "OfficeHome": 65,
    "TerraIncognita": 10,
    "DomainNet": 345,
}

_UAP_ITERS = {
    "RotatedMNIST": 3000,
    "ColoredMNIST": 3000,
}


_UAP_LRS = {
    "RotatedMNIST": 0.01,
    "ColoredMNIST": 0.01,
}

_UAT_LRS = {
    "RotatedMNIST": 0.01,
    "ColoredMNIST": 0.01,
}

    
def _get_FGSM_hparams(dataset, train):
    assert dataset in _dataset_eps, f"attack for dataset {dataset} is not implemented"
    eps = _dataset_eps[dataset]
    return {'eps': eps}
    
    
def _get_PGD_hparams(dataset, train):
    assert dataset in _dataset_eps, f"attack for dataset {dataset} is not implemented"
    steps = 10 if train else 20
    eps = _dataset_eps[dataset]
    return {'eps': eps, 'alpha': eps / 4, 'steps': steps, 'random_start': True}

def _get_TPGD_hparams(dataset, train=True):
    assert train, "The attack TPGD is only used for adversarial training but not adversarial attack"
    assert dataset in _dataset_eps, f"attack for dataset {dataset} is not implemented"
    steps = 10 if train else 20
    eps = _dataset_eps[dataset]
    return {'eps': eps, 'alpha': eps / 4, 'steps': steps}

def _get_AutoAttack_hparams(dataset, train):
    assert dataset in _dataset_eps, f"attack for dataset {dataset} is not implemented"
    eps = _dataset_eps[dataset]
    n_classes = _dateset_classes[dataset]
    assert n_classes > 2, f"AutoAttack does not support tasks with class number less than 3, the current class number is {n_classes}"
    version = 'standard' if n_classes > 3 else 'rand'
    return {'norm': 'Linf', 'eps': eps, 'version': version, 'n_classes': n_classes}

def _get_FastUAP_hparams(dataset, train=True):
    eps = _dataset_eps[dataset]
    raise NotImplementedError

def _get_SimpleUAP_hparams(dataset, train=True):
    eps = _dataset_eps[dataset]
    n_iters = _UAP_ITERS[dataset]
    lr = _UAP_LRS[dataset]
    return {"eps": eps, "lr": lr, "n_iters": n_iters}

def _get_uat_CUAT_hparams(dataset):
    assert dataset in _UAT_LRS, f"attack for dataset {dataset} is not implemented"
    lr = _UAT_LRS[dataset]
    eps = _dataset_eps[dataset]
    return {'lr': lr, 'eps': eps}

def _get_uat_DUAT_hparams(dataset):
    assert dataset in _UAT_LRS, f"attack for dataset {dataset} is not implemented"
    eps = _dataset_eps[dataset]
    return {'eps': eps}