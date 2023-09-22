"""
Resweep to run the retrain.py file to retrain the model with the selected hyper-parameters
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

import tqdm
import shlex

class RetrainJob:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'
    MOVE = 'Move'

    def __init__(self, train_args):
        self.output_dir = train_args['output_dir']
        self.train_args = copy.deepcopy(train_args)
        self.selection_method = self.train_args['selection_method']
        self.train_args.pop('checkpoint_freq')
        self.train_args.pop('skip_model_save')
        self.train_args.pop('save_model_every_checkpoint')
        self.train_args.pop('hparams') # we need to pop hparams out, other wise when it is None, it will set the cmd parameter to be a string "None"
        
        if os.path.exists(os.path.join(self.output_dir, '{}_done'.format(self.selection_method))):
            self.state = RetrainJob.DONE
        elif self.selection_method == 'OracleSelectionMethod':
            self.state = RetrainJob.MOVE
        elif os.path.exists(self.output_dir):
            self.state = RetrainJob.INCOMPLETE
        else:
            self.state = RetrainJob.NOT_LAUNCHED
            
        if self.state != RetrainJob.MOVE:
            command = ['python', '-m', 'domainbed.scripts.retrain']
            for k, v in sorted(self.train_args.items()):
                if isinstance(v, list):
                    v = ' '.join([str(v_) for v_ in v])
                elif isinstance(v, str):
                    v = shlex.quote(v)
                command.append(f'--{k} {v}') # add an argument and its value to form a parameter of a running command
            self.command_str = ' '.join(command) # get a command string to run
        else:
            source_path = os.path.join(self.output_dir, "model.pkl")
            target_path = os.path.join(self.output_dir, "{}_model.pkl".format(self.selection_method))
            done_path = os.path.join(self.output_dir, "{}_done".format(self.selection_method))
            self.command_str = f'cp {source_path} {target_path} && echo \"{self.selection_method}_done\" > {done_path}'
            

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        train_commands = [job.command_str for job in jobs if job.state != RetrainJob.MOVE]
        move_commands = [job.command_str for job in jobs if job.state == RetrainJob.MOVE]
        for move_command in move_commands:
            subprocess.Popen(move_command, shell=True)
        
        launcher_fn(train_commands)
        print(f'Launched {len(jobs)} retrain jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def make_args_list(sweep_base_dir, selection_methods, dataset_names, algorithms):
    assert os.path.exists(sweep_base_dir), "The sweep_base_dir does not exist!!!"
    args_list = []
    for dataset in dataset_names:
        dataset_hparam_dir = os.path.join(sweep_base_dir, "best_hparams", "{}".format(dataset))
        assert os.path.exists(dataset_hparam_dir), "The hyper-parameters directory of dataset {} does not exist".format(dataset)
        dataset_hparam_path = os.path.join(dataset_hparam_dir, "hparams.json")
        assert os.path.exists(dataset_hparam_path), "The hyper-parameters file of dataset {} does not exist".format(dataset)
        # load the hyper-parameter file
        with open(dataset_hparam_path, "r") as f:
            dataset_hparams = json.load(f)
        
        for selection_method in selection_methods:
            # assert selection_method.__name__ in dataset_hparams, "The model selection method {} is not in the hparam file".format(selection_method.__name__), allow it
            if selection_method.__name__ not in dataset_hparams:
                print("warning !!! The model selection method {} is not in the hparam file".format(selection_method.__name__))
                continue
            sm_hparams = dataset_hparams[selection_method.__name__]
            for algorithm in algorithms:
                # assert algorithm in sm_hparams, "Algorithm {} is not in the hparam file".format(algorithm), now allow it
                if algorithm not in sm_hparams:
                    print("warning !!! Algorithm {} is not in the hparam file".format(algorithm))
                    continue 
                alg_hparams = sm_hparams[algorithm]
                for ev_hparams in alg_hparams: # The hparams of some test environment
                    run_acc, out_dir, args, hparams = ev_hparams
                    args['steps'] = run_acc['step'] + 1
                    args['selection_method'] = selection_method.__name__
                    args_list.append(args)
    return args_list

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
    parser.add_argument('--selection_methods', nargs='+', type=str, default=model_selection.SELECT_METHODS)
    parser.add_argument('--sweep_base_dir', type=str, default="./sweep/output")
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()
    # args = parser.parse_args(['launch', '--datasets', 'ColoredMNIST', 'RotatedMNIST',
    #                           '--command_launcher', 'multi_gpu',
    #                           '--selection_methods', 'OracleSelectionMethod', 'IIDAccuracySelectionMethod',
    #                           ])
    
    
    ########## remember to transfer the selection_methods to the classes rather than the str before inputing into the make_args_list method
    sm_vars = vars(model_selection)
    sms = [sm_vars[selection_method] for selection_method in args.selection_methods]

    args_list = make_args_list(
        sweep_base_dir=args.sweep_base_dir,
        selection_methods=sms,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
    ) # get all the arglists to run

    jobs = [RetrainJob(train_args) for train_args in args_list] # create all the running jobs

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} move, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == RetrainJob.DONE]),
        len([j for j in jobs if j.state == RetrainJob.MOVE]),
        len([j for j in jobs if j.state == RetrainJob.INCOMPLETE]),
        len([j for j in jobs if j.state == RetrainJob.NOT_LAUNCHED]))
    ) # print the number, done number, incomplete number and not launched number of the jobs

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == RetrainJob.INCOMPLETE or j.state == RetrainJob.MOVE]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        RetrainJob.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete': # do not allowed to delete
        raise NotImplementedError
        to_delete = [j for j in jobs if j.state == RetrainJob.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        RetrainJob.delete(to_delete)
