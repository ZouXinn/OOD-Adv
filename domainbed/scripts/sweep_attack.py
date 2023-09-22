"""
Sweep to run the attack_models.py file to attack the models trained with the selected hyper-parameters
"""

import argparse
import copy
import getpass
import hashlib
import json
import os, sys
import random
import shutil
import time
import uuid


import numpy as np
import torch


cwd = os.getcwd()
sys.path.append(cwd)
from domainbed import datasets, model_selection
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers
import subprocess
import torchattacks

import tqdm
import shlex

class AttackJob:
    NO_MODEL = 'No Model'
    ATTACK = 'Attack'
    DONE = 'Done'
    INCOMPLETE = 'Incomplete' # no "done" file
    NOT_SELECT = 'Not Select' # no "selection_done" file

    def __init__(self, attack_args):
        self.model_dir = attack_args['model_dir']
        self.attack = attack_args['attack']
        self.selection_method = attack_args['selection_method']
        self.train_method = attack_args['train_method']
        self.attack_args = copy.deepcopy(attack_args)

        work_dir = os.path.join(self.model_dir, "attacks", self.selection_method)
        done_path = os.path.join(work_dir, f"{self.attack}_done")
        ## The out_path, err_path and result_path are used when we want to sweep attack
        self.out_path = os.path.join(work_dir, f"{self.attack}_out.txt")
        self.err_path = os.path.join(work_dir, f"{self.attack}_err.txt")
        self.result_path = os.path.join(work_dir, f"{self.attack}_results.json")
        #### we will run jobs with state DONE
        
        ################# need to consider whether the model selection is done for AT and UAT models
        if os.path.exists(done_path):
            self.state = AttackJob.DONE # which means that the attack is done
        elif os.path.exists(self.out_path) or os.path.exists(self.err_path) or os.path.exists(self.result_path):
            self.state = AttackJob.INCOMPLETE
        elif self.train_method == "ST":
            if os.path.exists(os.path.join(self.model_dir, f"{self.selection_method}_done")):
                self.state = AttackJob.ATTACK # which mean the model that will be attacked is prepared
            else:
                self.state = AttackJob.NO_MODEL
        else: # AT
            if os.path.exists(os.path.join(self.model_dir, "selection_done")):
                self.state = AttackJob.ATTACK
            elif os.path.exists(os.path.join(self.model_dir, "done")):
                self.state = AttackJob.NOT_SELECT
            else:
                self.state = AttackJob.NO_MODEL # which mean that the attacked model is not prepared
        
        # create the running command
        command = ['python', '-m', 'domainbed.scripts.attack_models']
        for k, v in sorted(self.attack_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}') # add an argument and its value to form a parameter of a running command
        self.command_str = ' '.join(command) # get a command string to run
            

    def __str__(self):
        job_info = (self.attack_args['attack'],
            self.attack_args['model_dir'],
            self.attack_args['selection_method'])
        return '{}: {} {}'.format(
            self.state,
            self.model_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.model_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        
        print(f'Launched {len(jobs)} attack jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            if os.path.exists(job.out_path):
                os.remove(job.out_path)
            if os.path.exists(job.err_path):
                os.remove(job.err_path)
            if os.path.exists(job.result_path):
                os.remove(job.result_path)
        print(f'Deleted {len(jobs)} jobs!')
    

def make_args_list(sweep_base_dir, selection_methods, dataset_names, attacks, algorithms, train_methods, AT_methods):
    assert os.path.exists(sweep_base_dir), "The sweep_base_dir does not exist!!!"
    args_list = []
    for dataset in dataset_names:
        dataset_hparam_dir = os.path.join(sweep_base_dir, "best_hparams", "{}".format(dataset))
        assert os.path.exists(dataset_hparam_dir), "The hyper-parameters directory of dataset {} does not exist".format(dataset)
        dataset_hparam_path = os.path.join(dataset_hparam_dir, "hparams.json") # the result json file path of the dataset hyper-parameter selection
        assert os.path.exists(dataset_hparam_path), "The hyper-parameters file of dataset {} does not exist".format(dataset)
        # load the hyper-parameter file
        with open(dataset_hparam_path, "r") as f:
            dataset_hparams = json.load(f)
        
        for selection_method in selection_methods:
            # assert selection_method.__name__ in dataset_hparams, "The model selection method {} is not in the hparam file".format(selection_method.__name__), allow it
            if selection_method.__name__ not in dataset_hparams:
                print("warning !!! The model selection method {} is not in the hparam file".format(selection_method.__name__))
                continue
            sm_hparams = dataset_hparams[selection_method.__name__] # the hparam results for the selection_method
            for algorithm in algorithms:
                # assert algorithm in sm_hparams, "Algorithm {} is not in the hparam file".format(algorithm), now allow it
                if algorithm not in sm_hparams:
                    print("warning !!! Algorithm {} is not in the hparam file".format(algorithm))
                    continue 
                alg_hparams = sm_hparams[algorithm] # the hparam results for the given algorithm under the selection_method
                for ev_hparams in alg_hparams: # The hparams of some test environment
                    run_acc, out_dir, args, hparams = ev_hparams
                    for attack in attacks:
                        if f'_get_{attack}_hparams' not in vars(hparams_registry):
                            print("warning !!! Attack {} is not supported now".format(attack))
                            continue
                        
                        # The attack_models.py file need model_dir, selection_method, train_method and attack, but in order to get the directory, we also need the AT_methods to make sure the directory of the attacks
                        for train_method in train_methods:
                            if train_method == "ST":
                                model_dir = out_dir
                                args_list.append({
                                    "model_dir": model_dir,
                                    "selection_method": selection_method.__name__,
                                    "train_method": train_method,
                                    "attack": attack
                                })
                            else:
                                train_methods_AT_methods = set(AT_DICT[train_method]) & set(AT_methods)
                                for AT_method in train_methods_AT_methods:
                                    model_dir = os.path.join(out_dir, f"{train_method}_{AT_method}", selection_method.__name__)
                                    args_list.append({
                                        "model_dir": model_dir,
                                        "selection_method": selection_method.__name__,
                                        "train_method": train_method,
                                        "attack": attack
                                    })
    return args_list

AT_DICT = {
    "AT": ["PGD", "TPGD"],
    "CUAT": ["CUAT"],
    "DUAT": [],
} # The supported AT types and AT methods

def check_AT_methods_AT_types(args):
    r'''
    @function: Check whether the args.AT_types and args.AT_methods are valid
    '''
    valid_dict = {}
    for k, v in AT_DICT.items():
        for method in v:
            valid_dict[method] = k
    
    for method in args.AT_methods:
        assert method in valid_dict.keys() and valid_dict[method] in args.train_methods

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--attacks', nargs='+', type=str, default=vars(torchattacks)['__all__'])
    parser.add_argument('--selection_methods', nargs='+', type=str, default=model_selection.SELECT_METHODS)
    parser.add_argument('--sweep_base_dir', type=str, default="./test_output")
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--train_methods', nargs='+', type=str, default=["ST", "AT", "CUAT", "DUAT"], help="chose what kind of trained models to attack")
    parser.add_argument('--AT_methods', nargs='+', type=str, default=["PGD", "TPGD", "CUAT", "DUAT"], help="the adversarial training methods for the adversarially trained models. This does not work for standard trained models")
    args = parser.parse_args()
    if "AT" in args.train_methods or "CUAT" in args.train_methods or "DUAT" in args.train_methods:
        check_AT_methods_AT_types(args)
    
    ########## remember to transfer the selection_methods to the classes rather than the str before inputing into the make_args_list method
    sm_vars = vars(model_selection)
    sms = [sm_vars[selection_method] for selection_method in args.selection_methods] # classes of the selection methods

    args_list = make_args_list(
        sweep_base_dir=args.sweep_base_dir,
        selection_methods=sms,
        dataset_names=args.datasets,
        attacks = args.attacks,
        algorithms=args.algorithms,
        train_methods=args.train_methods,
        AT_methods=args.AT_methods
    ) # get all the arglists to run

    jobs = [AttackJob(attack_args) for attack_args in args_list] # create all the running jobs

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} to be attack, {} incomplete, {} not selected, {} no model.".format(
        len(jobs),
        len([j for j in jobs if j.state == AttackJob.DONE]),
        len([j for j in jobs if j.state == AttackJob.ATTACK]),
        len([j for j in jobs if j.state == AttackJob.INCOMPLETE]),
        len([j for j in jobs if j.state == AttackJob.NOT_SELECT]),
        len([j for j in jobs if j.state == AttackJob.NO_MODEL]))
    ) # print the number, done number, incomplete number and not launched number of the jobs

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == AttackJob.ATTACK]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        AttackJob.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete': # do not allowed to delete
        ## remember to carefully read the delete method of the job class and then carefully modify them according to your requirement 
        to_delete = [j for j in jobs if j.state == AttackJob.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        AttackJob.delete(to_delete)
