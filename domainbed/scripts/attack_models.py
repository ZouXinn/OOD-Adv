'''
Use adversarial attack to attack the standard trained models, and show the results
1. load the trained model
2. attack the model
3. save the result
'''

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torchattacks
import hashlib

cwd = os.getcwd()
sys.path.append(cwd)

from domainbed import datasets, model_selection
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


def check_args(args):
    assert args.attack in vars(torchattacks)['__all__'], f"Attack method {args.attack} not implemented"
    assert args.selection_method in model_selection.SELECT_METHODS, f"Selection_method {args.selection_method} is not implemented"
    model_done_path = os.path.join(args.model_dir, f"{args.selection_method}_done")
    seletction_done_path = os.path.join(args.model_dir, "selection_done")
    if args.train_method == "ST":
        assert os.path.exists(model_done_path), f"The {model_done_path} does not exists, which mean the model is not done now, please finish training model first"
    else:
        assert os.path.exists(seletction_done_path), f"The {seletction_done_path} does not exists, which mean the model is not selected now, please finish selecting model first"
    

def check_alg_args(alg_args):
    raise NotImplementedError

def get_run_id_str(args, alg_args, attack_hparams):
    '''
    check whether the running is the same as that of item's, if the same, return true
    '''
    dump_dict = {
        "attack": args.attack,
        "dataset": alg_args["dataset"],
        "algorithm": alg_args["algorithm"],
        "selection_method": args.selection_method,
        "attack_hparams": attack_hparams
    }
    id_str = json.dumps(dump_dict, sort_keys=True)
    id_str_hash = hashlib.md5(id_str.encode('utf-8')).hexdigest()
    return id_str_hash

def run_unique():
    raise NotImplementedError
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    ####### model related
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--selection_method', type=str, default="IIDAccuracySelectionMethod")
    parser.add_argument('--train_method', choices=['ST', 'AT', 'CUAT', 'DUAT'], default="ST", help='whether the model is standard trained or adversarially trained? It decides the filename of the model')
    
    ####### dataset related

    ####### attack related
    parser.add_argument('--attack', type=str, default='FGSM')
    parser.add_argument('--num_steps', type=int, default=None,
        help='Number of steps to attack. Default is attack-dependent.')
    
    
    
    ####### Others
    parser.add_argument('--std_mode', type=str, default='w')
    ####### output related
    
    args = parser.parse_args()
    check_args(args)
    
    
    
    ####################################
    #### redirect output
    output_dir = os.path.join(args.model_dir, "attacks", args.selection_method)
    os.makedirs(output_dir, exist_ok=True)
    
  
    if args.train_method == "ST":
        algorithm_path = os.path.join(args.model_dir, f"{args.selection_method}_model.pkl")
    else:
        algorithm_path = os.path.join(args.model_dir, "model.pkl")
    
    # load model
    state_dict = torch.load(algorithm_path)
    algorithm_args = state_dict['args']
    algorithm_hparam = state_dict['model_hparams']
    algorithm_dict = state_dict['model_dict']
    model_input_shape = state_dict['model_input_shape']
    model_num_classes = state_dict['model_num_classes']
    model_num_domains = state_dict['model_num_domains']
    
    if args.train_method != "ST":
        selected_result_path = os.path.join(args.model_dir, "selected_results.jsonl")
        records = []
        try:
            with open(selected_result_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            assert True, "reading the {} file failed".format(selected_result_path)
        algorithm_args = records[0]["st_args"]
    
    attack_hparams = hparams_registry.attack_hparams(args.attack, algorithm_args['dataset'])
    if "num_steps" in attack_hparams and args.num_steps:
        attack_hparams["num_steps"] = args.num_steps
    
    done_path = os.path.join(output_dir, f'{args.attack}_done')
    result_path = os.path.join(output_dir, f"{args.attack}_results.json")
    
    if os.path.exists(done_path):
        print("attack already done, exiting...")
        exit(0)
    
    
    # std redirection
    sys.stdout = misc.Tee(os.path.join(output_dir, f'{args.attack}_out.txt'), mode=args.std_mode)
    sys.stderr = misc.Tee(os.path.join(output_dir, f'{args.attack}_err.txt'), mode=args.std_mode)
    
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
        
    print('Algorithm args:')
    for k, v in sorted(algorithm_args.items()):
        print('\t{}: {}'.format(k, v))


    if algorithm_args['hparams_seed'] == 0:
        hparams = hparams_registry.default_hparams(algorithm_args['algorithm'], algorithm_args['dataset'])
    else:
        hparams = hparams_registry.random_hparams(algorithm_args['algorithm'], algorithm_args['dataset'],
            misc.seed_hash(algorithm_args['hparams_seed'], algorithm_args['trial_seed']))
    if algorithm_args['hparams']:
        hparams.update(json.loads(algorithm_args['hparams']))
        
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
        
    
    print('Attack hparams:')
    for k, v in sorted(attack_hparams.items()):
        print('\t{}: {}'.format(k, v))
        
    
    

    random.seed(algorithm_args['seed'])
    np.random.seed(algorithm_args['seed'])
    torch.manual_seed(algorithm_args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #### load dataset
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    if algorithm_args['dataset'] in vars(datasets):
        dataset = vars(datasets)[algorithm_args['dataset']](algorithm_args['data_dir'],
            algorithm_args['test_envs'], hparams) # 
    else:
        raise NotImplementedError
    
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*algorithm_args['holdout_fraction']),
            misc.seed_hash(algorithm_args['trial_seed'], env_i))

        if env_i in algorithm_args['test_envs']: # split the uda split for the test envs
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*algorithm_args['uda_holdout_fraction']),
                misc.seed_hash(algorithm_args['trial_seed'], env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if algorithm_args['task'] == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)] # the test envs, use all the splits
    
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)] # no weights when evaluating
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
    
    #### load model
    
    
    algorithm_class = algorithms.get_algorithm_class(algorithm_args['algorithm'])
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(algorithm_args['test_envs']), hparams)
    
    
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
    
    algorithm.to(device)
    algorithm.eval()
    #### load attacker
    attack_class = vars(torchattacks)[args.attack] # get the class of the attack
    
    
    attack = attack_class(algorithm, **attack_hparams)
    if "MNIST" not in algorithm_args['dataset']:
        attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    ##### begin to collect the accuracy and the adversarial accuracy
    results = {}
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals: # evaluate on every environments and record results
        acc = misc.clean_accuracy(algorithm, loader, weights, device)
        adv_acc = misc.adversarial_accuracy(algorithm, attack, loader, weights, device)
        results[name+'_clean_acc'] = acc
        results[name+'_adv_acc'] = adv_acc

    save_dict = {
        "attack": args.attack,
        "dataset": algorithm_args['dataset'],
        "algorithm": algorithm_args['algorithm'],
        "selection_method": args.selection_method,
        "attack_hparams": attack_hparams,
        "test_envs": sorted(algorithm_args['test_envs']), # use the sorted one to ensure the test_envs to be ordered
        'attack_results': results
    }

    json_str = json.dumps(save_dict, indent=4)
    with open(result_path, 'w') as json_file:
        json_file.write(json_str)

    with open(done_path, 'w') as f:
        f.write('attack done')