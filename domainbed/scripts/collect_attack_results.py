'''
This file collects the results of the adversarial attacks for the models trained from the optimal hyper-parameters.
Note: this script does not to scan all the files of trained models
'''
import argparse
import copy
import getpass
import hashlib
import json
import os, sys
from statistics import mean
import random
import shutil
import time
import uuid

import numpy as np
import torch
import torchattacks

cwd = os.getcwd()
sys.path.append(cwd)
from domainbed import datasets, model_selection
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed.lib.query import Q

from domainbed.scripts.collect_results import print_table

import tqdm
import shlex

def make_args_list(sweep_base_dir, selection_methods, dataset_names, attacks, algorithms, train_method, AT_methods):
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
                        if train_method == "ST":
                            args_list.append({"model_dir": out_dir,
                                            "selection_method": selection_method.__name__,
                                            "attack": attack,
                                            "algorithm": args["algorithm"],
                                            "dataset":args["dataset"],
                                            "train_method": train_method
                                            })
                        else:
                            for AT_method in AT_methods:
                                apd_dir = f"{train_method}_{AT_method}"
                                args_list.append({"model_dir": os.path.join(out_dir, apd_dir, selection_method.__name__),
                                            "selection_method": selection_method.__name__,
                                            "attack": attack,
                                            "algorithm": args["algorithm"],
                                            "dataset":args["dataset"],
                                            "train_method": train_method,
                                            "AT_method": AT_method
                                            })
    return args_list


def load_attack_records(arg_list, anchors_per_class_per_envs):
    records = []
    for arg in arg_list:
        attack = arg["attack"]
        if attack in CUAPS: # classwise universal adversarial attack
            for apcpe in anchors_per_class_per_envs:
                apcpe_str = f"{apcpe}" if apcpe else "all"
                record_dir = os.path.join(arg['model_dir'], "cuniversal_attacks", arg['selection_method'], apcpe_str)
                if not os.path.exists(record_dir):
                    print(f"Warning !!! Path {record_dir} does not exist... Skip it.")
                    continue
                done_path = os.path.join(record_dir, f"{attack}_done")
                if not os.path.exists(done_path):
                    print(f"Warning !!! Attack {attack} is not finished right now.")
                    continue
                json_path = os.path.join(record_dir, f"{attack}_results.json")
                if not os.path.exists(json_path):
                    print(f"Warning !!! The result file of Attack {attack} does not exist.")
                    continue
                with open(json_path, "r") as f:
                    attack_results = json.load(f)
                attack_results["anchors_per_class_per_env"] = apcpe_str # add the cpcpe property
                if "AT_method" in arg: # add the AT_method property
                    attack_results["AT_method"] = arg["AT_method"]
                records.append(attack_results)
        else: # adversarial attack
            record_dir = os.path.join(arg['model_dir'], "attacks", arg['selection_method'])
            if not os.path.exists(record_dir):
                print(f"Warning !!! Path {record_dir} does not exist... Skip it.")
                continue
            done_path = os.path.join(record_dir, f"{attack}_done")
            if not os.path.exists(done_path):
                print(f"Warning !!! Attack {attack} is not finished right now.")
                continue
            json_path = os.path.join(record_dir, f"{attack}_results.json")
            if not os.path.exists(json_path):
                print(f"Warning !!! The result file of Attack {attack} does not exist.")
                continue
            with open(json_path, "r") as f:
                attack_results = json.load(f)
            if "AT_method" in arg: # add the AT_method property
                attack_results["AT_method"] = arg["AT_method"]
            records.append(attack_results) # Add this record
    return Q(records)

def print_attack_result_tables_ST(records: Q, selection_method: str, args): # it chooses the in_clean acc and in_adv acc as the target acc
    selection_records = records.filter(lambda x: x["selection_method"] == selection_method)
    if len(selection_records) == 0:
        print(f"No result for selection method {selection_method}")
        return None
    all_alg_names = Q(records).select("algorithm").unique()
    all_alg_names = ([n for n in algorithms.ALGORITHMS if n in all_alg_names] +
        [n for n in all_alg_names if n not in algorithms.ALGORITHMS]) # get all possible algorithm names
    
    all_table = {}
    for attack in args.attacks:
        all_table[attack] = [[None for _ in [*args.datasets, "Avg"]] for _ in all_alg_names]
    
    for d, dataset in enumerate(args.datasets):
        if args.latex:
            print()
            print("\\subsection{{{}}}".format(dataset))
        s_dataset_records = selection_records.filter(lambda x: x["dataset"] == dataset)
        if len(s_dataset_records) == 0:
            print(f"No result for dataset {dataset}\n")
            continue
        eps = s_dataset_records[0]["attack_hparams"]["eps"]
        
        for attack in args.attacks:
            if args.latex:
                print()
                print("\\subsubsection{{{} Attack}}".format(attack))
            
            s_d_attack_records = s_dataset_records.filter(lambda x: x["attack"] == attack)
            if len(s_d_attack_records) == 0:
                print(f"No result for attack {attack}\n")
                continue
        
            alg_names = Q(records).select("algorithm").unique()
            alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
                [n for n in alg_names if n not in algorithms.ALGORITHMS]) # get all possible algorithm names
            
            test_envs = range(datasets.num_environments(dataset)) # get all possible test envs for the dataset
            table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
            
            for i, algorithm in enumerate(alg_names): # iterate through the algorithms
                s_d_a_algorithm_1tenv_records = s_d_attack_records.filter(lambda x: x["algorithm"] == algorithm).filter(lambda x: len(x["test_envs"]) == 1)
                std_accs = []
                adv_accs = []
                for j, test_env in enumerate(test_envs): # iterate through the test envs
                    s_d_a_a_test_env_records = s_d_a_algorithm_1tenv_records.filter(lambda x: test_env in x["test_envs"])
                    if len(s_d_a_a_test_env_records) == 0:
                        table[i][j] = 'X'
                    elif len(s_d_a_a_test_env_records) > 1:
                        table[i][j] = 'M' # multiple results, do not know which one to choose
                    else: # get the test env accuracy
                        std_key = f"env{test_env}_in_clean_acc"
                        adv_key = f"env{test_env}_in_adv_acc"
                        std_acc = s_d_a_a_test_env_records[0]["attack_results"][std_key]
                        adv_acc = s_d_a_a_test_env_records[0]["attack_results"][adv_key]
                        std_accs.append(std_acc)
                        adv_accs.append(adv_acc)
                        table[i][j] = "{:.1f} / {:.1f}".format(std_acc*100, adv_acc*100)
                # handle the avg acc
                ds_res = ()
                if len(std_accs) == len(test_envs):
                    std_avg = "{:.1f}".format(mean(std_accs)*100)
                    ds_res += (mean(std_accs)*100,)
                else:
                    std_avg = "X"
                    ds_res += (None,)
                if len(adv_accs) == len(test_envs):
                    adv_avg = "{:.1f}".format(mean(adv_accs)*100)
                    ds_res += (mean(adv_accs)*100,)
                else:
                    adv_avg = "X"
                    ds_res += (None,)
                table[i][-1] = f"{std_avg} / {adv_avg}"
                
                all_table[attack][i][d] = ds_res
            col_labels = [
                "Algorithm",
                *datasets.get_dataset_class(dataset).ENVIRONMENTS,
                "Avg"
            ] # the column labels used in the table
            header_text = f"Selection method: {selection_method}, Dataset: {dataset}, Attack {attack}, l-inf , epsilon = {eps}"
            print_table(table, header_text, alg_names, list(col_labels),
                colwidth=20, latex=args.latex) # print out the corresponding table
    for attack in args.attacks:
        attack_table = all_table[attack]
        for i in range(len(all_alg_names)):
            total_acc_std = 0
            flag_std = True
            total_acc_adv = 0
            flag_adv = True
            for j in range(len(args.datasets)):
                std_s_str = "X"
                adv_s_str = "X"
                if flag_std and attack_table[i][j][0] is not None:
                    total_acc_std += attack_table[i][j][0]
                else:
                    flag_std = False
                if flag_adv and attack_table[i][j][1] is not None:
                    total_acc_adv += attack_table[i][j][1]
                else:
                    flag_adv = False
                    
                if attack_table[i][j][0] is not None:
                    std_s_str = "{:.1f}".format(attack_table[i][j][0])
                if attack_table[i][j][1] is not None:
                    adv_s_str = "{:.1f}".format(attack_table[i][j][1])
                all_table[attack][i][j] = f"{std_s_str} / {adv_s_str}"
                
            if flag_std:
                std_str = "{:.1f}".format(total_acc_std/len(args.datasets))
            else:
                std_str =  "X"
            if flag_adv:
                adv_str = "{:.1f}".format(total_acc_adv/len(args.datasets))
            else:
                adv_str =  "X"
                
            all_table[attack][i][-1] = f"{std_str} / {adv_str}"
        ### print table
        col_labels = ["Algorithm", *args.datasets, "Avg"]
        header_text = f"Selection method: {selection_method}, Attack {attack}, l-inf , epsilon = {eps}, all datasets"
        print_table(all_table[attack], header_text, alg_names, list(col_labels),
            colwidth=20, latex=args.latex)

def print_attack_result_tables_ST_attack_all_in_one(records: Q, selection_method: str, args): # it chooses the in_clean acc and in_adv acc as the target acc
    selection_records = records.filter(lambda x: x["selection_method"] == selection_method)
    if len(selection_records) == 0:
        print(f"No result for selection method {selection_method}")
        return None
    all_alg_names = Q(records).select("algorithm").unique()
    all_alg_names = ([n for n in algorithms.ALGORITHMS if n in all_alg_names] +
        [n for n in all_alg_names if n not in algorithms.ALGORITHMS]) # get all possible algorithm names
    
    all_table = [[None for _ in [*args.datasets, "Avg"]] for _ in all_alg_names]
    
    for d, dataset in enumerate(args.datasets):
        if args.latex:
            print()
            print("\\subsection{{{}}}".format(dataset))
        s_dataset_records = selection_records.filter(lambda x: x["dataset"] == dataset)
        if len(s_dataset_records) == 0:
            print(f"No result for dataset {dataset}\n")
            continue
        eps = s_dataset_records[0]["attack_hparams"]["eps"]
    
        alg_names = Q(records).select("algorithm").unique()
        alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
            [n for n in alg_names if n not in algorithms.ALGORITHMS]) # get all possible algorithm names
        
        test_envs = range(datasets.num_environments(dataset)) # get all possible test envs for the dataset
        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        
        items_label = "clean"
        for attack in args.attacks:
            items_label += " / {}".format(attack)
        
        for i, algorithm in enumerate(alg_names): # iterate through the algorithms
            s_d_algorithm_1tenv_records = s_dataset_records.filter(lambda x: x["algorithm"] == algorithm).filter(lambda x: len(x["test_envs"]) == 1)
             
            std_accs = []
            adv_accs = [[] for _ in args.attacks]
            for j, test_env in enumerate(test_envs): # iterate through the test envs
                s_d_a_test_env_records = s_d_algorithm_1tenv_records.filter(lambda x: test_env in x["test_envs"])
                
                std_key = f"env{test_env}_in_clean_acc"
                if len(s_d_a_test_env_records) == 0:
                    table[i][j] = "X"
                elif len(s_d_a_test_env_records) > len(args.attacks):
                    table[i][j] = "M"
                else:
                    std_key = f"env{test_env}_in_clean_acc"
                    std_acc = s_d_a_test_env_records[0]["attack_results"][std_key]
                    std_accs.append(std_acc)
                    table[i][j] = "{:.1f}".format(std_acc*100)
                    
                for k in range(len(args.attacks)):
                    s_d_a_t_attack_env_records = s_d_a_test_env_records.filter(lambda x: x["attack"] == args.attacks[k])
                    if len(s_d_a_t_attack_env_records) == 0:
                        table[i][j] += " \ X"
            
                    elif len(s_d_a_t_attack_env_records) > 1:
                        table[i][j] += " \ M"
                    else:
                        adv_key = f"env{test_env}_in_adv_acc"
                        adv_acc = s_d_a_t_attack_env_records[0]["attack_results"][adv_key]
                        adv_accs[k].append(adv_acc)
                        table[i][j] += " / {:.1f}".format(adv_acc*100)
            # handle the avg acc
            all_table_item = []
            if len(std_accs) == len(test_envs):
                std_avg = "{:.1f}".format(mean(std_accs)*100)
                all_table_item.append(mean(std_accs)*100)
            else:
                std_avg = "X"
                all_table_item.append(None)
            adv_avgs = []
            for k in range(len(args.attacks)):
                if len(adv_accs[k]) == len(test_envs):
                    adv_avgs.append("{:.1f}".format(mean(adv_accs[k])*100))
                    all_table_item.append(mean(adv_accs[k])*100)
                else:
                    adv_avgs.append("X")
                    all_table_item.append(None)
            avg = std_avg
            for adv_avg in adv_avgs:
                avg += " / {}".format(adv_avg)
            table[i][-1] = avg
            
            
            all_table[i][d] = all_table_item
            
        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ] # the column labels used in the table
        header_text = f"Selection method: {selection_method}, Dataset: {dataset}, Attacks {args.attacks}, l-inf , epsilon = {eps}, " + items_label
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=args.latex) # print out the corresponding table
            
    for i in range(len(all_alg_names)):
        total_acc_std = 0
        flag_std = True
        total_acc_advs = [0 for _ in args.attacks]
        flag_advs = [True for _ in args.attacks]
        pass
        for d in range(len(args.datasets)):
            std_s_str = "X"
            adv_s_strs = ["X" for _ in args.attacks]
            if flag_std and all_table[i][d][0] is not None:
                total_acc_std += all_table[i][d][0]
            else:
                flag_std = False
                
            for k in range(len(args.attacks)):
                if flag_advs[k] and all_table[i][d][k+1] is not None:
                    total_acc_advs[k] += all_table[i][d][k+1]
                else:
                    flag_advs[k] = False
            
            if all_table[i][d][0] is not None:
                std_s_str = "{:.1f}".format(all_table[i][d][0])
            else:
                std_s_str = "X"
            item_str = std_s_str
            for k in range(len(args.attacks)):
                if all_table[i][d][k+1] is not None:
                    item_str += " / {:.1f}".format(all_table[i][d][k+1])
                else:
                    item_str += " / X"
            all_table[i][d] = item_str
            
        if flag_std:
            std_str = "{:.1f}".format(total_acc_std/len(args.datasets))
        else:
            std_str = "X"
        avg_str = std_str
        for k in range(len(args.attacks)):
            if flag_advs[k]:
                avg_str +=  "/ {:.1f}".format(total_acc_advs[k]/len(args.datasets))
            else:
                avg_str += " / X"
        all_table[i][-1] = avg_str

    ### print table
    col_labels = ["Algorithm", *args.datasets, "Avg"]
    header_text = f"Selection method: {selection_method}, Attacks {args.attacks}, l-inf , epsilon = {eps}, all datasets, " + items_label
    print_table(all_table, header_text, alg_names, list(col_labels),
        colwidth=20, latex=args.latex)

def print_attack_result_tables_AT(records: Q, selection_method: str, args, AT_method): # it chooses the in_clean acc and in_adv acc as the target acc
    selection_records = records.filter(lambda x: x["selection_method"] == selection_method)
    if len(selection_records) == 0:
        print(f"No result for selection method {selection_method}")
        return None
    selection_ATM_records = selection_records.filter(lambda x: x["AT_method"] == AT_method)
    if len(selection_ATM_records) == 0:
        print(f"No result for AT method {AT_method}")
    for dataset in args.datasets:
        if args.latex:
            print()
            print("\\subsection{{{}}}".format(dataset))
        s_dataset_records = selection_ATM_records.filter(lambda x: x["dataset"] == dataset)
        if len(s_dataset_records) == 0:
            print(f"No result for dataset {dataset}\n")
            continue
        eps = s_dataset_records[0]["attack_hparams"]["eps"]
        
        for attack in args.attacks:
            if args.latex:
                print()
                print("\\subsubsection{{{} Attack}}".format(attack))
                
            if attack in CUAPS:
                apcpe_strs = [f"{apcpe}" if apcpe else "all" for apcpe in args.anchors_per_class_per_envs]
                for apcpe_str in apcpe_strs:
                    if args.latex:
                        print()
                        print("\\textbf{{{} Anchors Attack}}".format(apcpe_str))
                    s_d_attack_records = s_dataset_records.filter(lambda x: x["attack"] == attack)
                    if len(s_d_attack_records) == 0:
                        print(f"No result for attack {attack}\n")
                        continue
                    s_d_attack_apcpe_records = s_d_attack_records.filter(lambda x: x["anchors_per_class_per_env"]==apcpe_str)
                    if len(s_d_attack_apcpe_records) == 0:
                        print(f"No results for anchors_per_class_per_env {apcpe_str}")
                        continue
                    alg_names = Q(records).select("algorithm").unique()
                    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
                        [n for n in alg_names if n not in algorithms.ALGORITHMS]) # get all possible algorithm names
                
                    test_envs = range(datasets.num_environments(dataset)) # get all possible test envs for the dataset
                    table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
                    
                    for i, algorithm in enumerate(alg_names): # iterate through the algorithms
                        s_d_a_algorithm_1tenv_records = s_d_attack_apcpe_records.filter(lambda x: x["algorithm"] == algorithm).filter(lambda x: len(x["test_envs"]) == 1)
                        std_accs = []
                        adv_accs = []
                        for j, test_env in enumerate(test_envs): # iterate through the test envs
                            s_d_a_a_test_env_records = s_d_a_algorithm_1tenv_records.filter(lambda x: test_env in x["test_envs"])
                            if len(s_d_a_a_test_env_records) == 0:
                                table[i][j] = 'X'
                            elif len(s_d_a_a_test_env_records) > 1:
                                table[i][j] = 'M' # multiple results, do not know which one to choose
                            else: # get the test env accuracy
                                std_key = f"env{test_env}_in_clean_acc"
                                adv_key = f"env{test_env}_in_adv_acc"
                                std_acc = s_d_a_a_test_env_records[0]["attack_results"][std_key]
                                adv_acc = s_d_a_a_test_env_records[0]["attack_results"][adv_key]
                                std_accs.append(std_acc)
                                adv_accs.append(adv_acc)
                                table[i][j] = "{:.1f} / {:.1f}".format(std_acc*100, adv_acc*100)
                        # handle the avg acc
                        if len(std_accs) == len(test_envs):
                            std_avg = "{:.1f}".format(mean(std_accs)*100)
                        else:
                            std_avg = "X"
                        if len(adv_accs) == len(test_envs):
                            adv_avg = "{:.1f}".format(mean(adv_accs)*100)
                        else:
                            adv_avg = "X"
                        table[i][-1] = f"{std_avg} / {adv_avg}"
                    col_labels = [
                        "Algorithm",
                        *datasets.get_dataset_class(dataset).ENVIRONMENTS,
                        "Avg"
                    ] # the column labels used in the table
                    header_text = f"Selection method: {selection_method}, Dataset: {dataset}, Attack {attack}, l-inf , epsilon = {eps}"
                    print_table(table, header_text, alg_names, list(col_labels),
                        colwidth=20, latex=args.latex) # print out the corresponding table
            else:
                s_d_attack_records = s_dataset_records.filter(lambda x: x["attack"] == attack)
                if len(s_d_attack_records) == 0:
                    print(f"No result for attack {attack}\n")
                    continue
            
                alg_names = Q(records).select("algorithm").unique()
                alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
                    [n for n in alg_names if n not in algorithms.ALGORITHMS]) # get all possible algorithm names
                
                test_envs = range(datasets.num_environments(dataset)) # get all possible test envs for the dataset
                table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
                
                for i, algorithm in enumerate(alg_names): # iterate through the algorithms
                    s_d_a_algorithm_1tenv_records = s_d_attack_records.filter(lambda x: x["algorithm"] == algorithm).filter(lambda x: len(x["test_envs"]) == 1)
                    std_accs = []
                    adv_accs = []
                    for j, test_env in enumerate(test_envs): # iterate through the test envs
                        s_d_a_a_test_env_records = s_d_a_algorithm_1tenv_records.filter(lambda x: test_env in x["test_envs"])
                        if len(s_d_a_a_test_env_records) == 0:
                            table[i][j] = 'X'
                        elif len(s_d_a_a_test_env_records) > 1:
                            table[i][j] = 'M' # multiple results, do not know which one to choose
                        else: # get the test env accuracy
                            std_key = f"env{test_env}_in_clean_acc"
                            adv_key = f"env{test_env}_in_adv_acc"
                            std_acc = s_d_a_a_test_env_records[0]["attack_results"][std_key]
                            adv_acc = s_d_a_a_test_env_records[0]["attack_results"][adv_key]
                            std_accs.append(std_acc)
                            adv_accs.append(adv_acc)
                            table[i][j] = "{:.1f} / {:.1f}".format(std_acc*100, adv_acc*100)
                    # handle the avg acc
                    if len(std_accs) == len(test_envs):
                        std_avg = "{:.1f}".format(mean(std_accs)*100)
                    else:
                        std_avg = "X"
                    if len(adv_accs) == len(test_envs):
                        adv_avg = "{:.1f}".format(mean(adv_accs)*100)
                    else:
                        adv_avg = "X"
                    table[i][-1] = f"{std_avg} / {adv_avg}"
                col_labels = [
                    "Algorithm",
                    *datasets.get_dataset_class(dataset).ENVIRONMENTS,
                    "Avg"
                ] # the column labels used in the table
                header_text = f"Selection method: {selection_method}, Dataset: {dataset}, Attack {attack}, l-inf , epsilon = {eps}"
                print_table(table, header_text, alg_names, list(col_labels),
                    colwidth=20, latex=args.latex) # print out the corresponding table


DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]
CUAPS = ["SimpleUAP"]
AT_DICT = {
    "AT": ["PGD", "TPGD"],
    "CUAT": ["CUAT"],
    "DUAT": [],
} # The supported AT types and AT methods

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True) # the dir of the training results, where we which contains sub folders such as path, best_params
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS) # the datasets which we would like to collect results
    parser.add_argument('--attacks', nargs='+', type=str, default=vars(torchattacks)['__all__'] + CUAPS) # the attacks which we would like to collect results
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS) # the algorithms which we would like to collect results
    parser.add_argument('--selection_methods', nargs='+', type=str, default=model_selection.SELECT_METHODS) # the selection_methods which we would like to collect results
    parser.add_argument("--latex", action="store_true") # whether to write to a latex file
    parser.add_argument("--train_method", choices=["ST", "AT", "CUAT", "DUAT"])
    parser.add_argument("--AT_methods", nargs='+', type=str, default=["PGD"], help="This only works when the train_method is not ST")
    parser.add_argument('--attack_all_in_one', action='store_true')
    parser.add_argument('--anchors_per_class_per_envs', nargs='+', type=int, default=[100, 200, 400, None], help="The number of data per_class_per_env used to train the CUAP, it only works for CUAP")
    
    args = parser.parse_args()
    # output directions
    assert os.path.exists(args.input_dir), f"The input_dir {args.input_dir} does not exist!"
    output_dir = os.path.join(args.input_dir, "attacks_results")
    os.makedirs(output_dir, exist_ok=True)
    
    sm_vars = vars(model_selection)
    sms = [sm_vars[selection_method] for selection_method in args.selection_methods] # classes of the selection methods
    
    args_list = make_args_list(
        sweep_base_dir=args.input_dir,
        selection_methods=sms,
        dataset_names=args.datasets,
        attacks = args.attacks,
        algorithms=args.algorithms,
        train_method=args.train_method,
        AT_methods=args.AT_methods,
    ) # get all the arglists to run
    print(f"There are totally {len(args_list)} args")
    records = load_attack_records(args_list, args.anchors_per_class_per_envs)
    print(f"There are totally {len(records)} records")
    
    for selection_method in args.selection_methods:
        if args.train_method == "ST":
            result_filename = f"{selection_method}_{args.train_method}"
            results_file = f"{result_filename}.tex" if args.latex else f"{result_filename}.txt"
            if args.attack_all_in_one:
                results_file = "all_in_one_" + results_file
            sys.stdout = misc.Tee(os.path.join(output_dir, results_file), "w")
            if args.latex:
                print("\\documentclass{article}")
                print("\\usepackage{booktabs}")
                print("\\usepackage{adjustbox}")
                print("\\begin{document}")
                print("\\section{Attack results for selected models}")
                print("% Total records:", len(records))
            else:
                print("Total records:", len(records))
            if args.attack_all_in_one:
                print_attack_result_tables_ST_attack_all_in_one(records, selection_method, args)
            else:
                print_attack_result_tables_ST(records, selection_method, args)
            if args.latex:
                print("\\end{document}")
        else: # AT, CUAT, DUAT
            for AT_method in args.AT_methods:
                if AT_method in AT_DICT[args.train_method]:
                    result_filename = f"{selection_method}_{args.train_method}_{AT_method}"
                    results_file = f"{result_filename}.tex" if args.latex else f"{result_filename}.txt"
                    sys.stdout = misc.Tee(os.path.join(output_dir, results_file), "w")
                    if args.latex:
                        print("\\documentclass{article}")
                        print("\\usepackage{booktabs}")
                        print("\\usepackage{adjustbox}")
                        print("\\begin{document}")
                        print("\\section{Attack results for selected models}")
                        print("% Total records:", len(records))
                    else:
                        print("Total records:", len(records))
                    print_attack_result_tables_AT(records, selection_method, args, AT_method)
                    if args.latex:
                        print("\\end{document}")
