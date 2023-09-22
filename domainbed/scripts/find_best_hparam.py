'''
think about how to handle different trial_seeds before writing codes, I just choose to write a single method to realize the case with just only one trial. For the case with many trials, we can think about whether to change the misc.seed_hash(args.hparams_seed, args.trial_seed) to misc.seed_hash(args.hparams_seed) in train.py at the code to use hparams = hparams_registry.random_hparams when I really need to run many trials. Here I recommend to change it and take the average of a choice of hyper-parameter to select the best choice of hyper-parameters, although it will need a lot of changes in the code.


#### Note that after choosing the best params, we should retrain the models by the algorithems with the best hyper-parameters
'''


import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
from select import select
import sys
from xmlrpc.client import boolean

import numpy as np
import tqdm

cwd = os.getcwd()
sys.path.append(cwd)

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
from domainbed.scripts.collect_results import print_table
import warnings


# def str2seletcion_method(method: str) -> model_selection.SelectionMethod:
#     '''
    
#     '''
#     raise NotImplementedError

def find_best_params_of_multi_trials(dataset: str, model_selection_method: model_selection.SelectionMethod, records: Q, latex: bool):
    '''
    @ Coder: Xin Zou
    @ Function: find the best hparams of a particular (dataset, algorithm) under a particular model selection method
    @ Parameters:
        @ dataset: the string specifying the dataset
        @ model_selection_method: the model selection method
        @ records: the records obtained from the function "reporting.load_records"
        @ latex: whether to print the table in a latex form
    @ return: dict (table? whether to just print it? do we need to return this?)
        the dict(w.r.t. algorithm) of the list (w.r.t. test_env) of best hyper-parameters together with the run_acc with the highest averaged (the average is w.r.t. the trial_seeds) validation accuracy, i.e. {"alg_name":[test_env_idx]}, where each element (return[alg_name][test_env]) is a tuple of (run_acc, output_dir, args, hparams) where the output_dir is the directory to 
    '''
    raise NotImplementedError

def find_best_params_of_1trial_alg_env_tab(dataset: str, model_selection_method: model_selection.SelectionMethod, records: Q, latex: bool):
    '''
    @ Coder: Xin Zou
    @ Function: find the best hparams of a particular (dataset, algorithm) under a particular model selection method
    @ Parameters:
        @ dataset: the string specifying the dataset
        @ model_selection_method: the model selection method
        @ records: the records obtained from the function "reporting.load_records"
        @ latex: whether to print the table in a latex form
    @ return: dict (table? whether to just print it? do we need to return this?)
        the dict(w.r.t. algorithm) of the list (w.r.t. test_env) of best hyper-parameters together with the run_acc with the highest averaged (the average is w.r.t. the trial_seeds) validation accuracy, i.e. {"alg_name":[test_env_idx]}, where each element (return[alg_name][test_env]) is a tuple of (run_acc, output_dir, args, hparams) where the output_dir is the directory to 
        
    '''
    grouped_records = reporting.get_grouped_records(records) # the elements are of structure {"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e, "records": Q(r)} where Q(r) is the list of records that have the same (trial_seed, dataset, algorithm. *a* test_env)
    grouped_records_aah = grouped_records.map(lambda group: {**group, "runacc_args_hparams": model_selection_method.sweep_acc_args_hparams_1trial(group['records'])})
    grouped_records_aah_filtered = grouped_records_aah.filter(lambda g: g["runacc_args_hparams"] is not None)
    
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] + [n for n in alg_names if n not in algorithms.ALGORITHMS])
    
    
    test_envs = range(datasets.num_environments(dataset)) # the all posible test envs
    dict_result = collections.defaultdict(lambda: [])
    
    table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        test_accs = []
        for j, test_env in enumerate(test_envs):
            trial_results = (grouped_records_aah_filtered.filter_equals("dataset, algorithm, test_env", 
                            (dataset, algorithm, test_env)).select("runacc_args_hparams"))
            assert len(trial_results) <= 1, "the number of trial results must be no more than 1, but it is {len(trial_results)}"
            if len(trial_results) == 1:
                test_accs.append(trial_results[0][0]['test_acc'])
                run_acc, args, hparams = trial_results[0]
                dict_result[algorithm].append((run_acc, args["output_dir"], args, hparams))
                table[i][j] = "{:.1f}".format(trial_results[0][0]['test_acc'] * 100)
            else: # The length is 0, which means there is no results for this environment
                test_accs.append(None) # append None for table results
                # do not append anything for the best_hparams
                table[i][j] = "X" # write "X" for this environment
        # handle the average term
        if None in test_accs:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(test_accs) * 100 / len(test_accs))
    col_labels = ["Algorithm", *datasets.get_dataset_class(dataset).ENVIRONMENTS, "Avg"] # the column labels used in the table
    header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
    print_table(table, header_text, alg_names, list(col_labels), colwidth=20, latex=latex)
    return dict_result
    ######### maybe we can use the hparams_accs() method to do hyper-parameter selection or even change the sweep_acc() method, just let the sweep_acc() to return addtionally the hparams???
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the the best parameters of the training models for a particular dataset and algorithm")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument("--latex", action="store_true")
    
    args = parser.parse_args()
    # args = parser.parse_args(["--input_dir", "./sweep/output/path", 
    #                           "--dataset", "RotatedMNIST",]) # remember to consider different test_env
    
    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        # model_selection.LeaveOneOutSelectionMethod, # there is no results of len(test_envs) == 2, so can not use it now
        # model_selection.OracleSelectionMethod,
    ]
    
    up_dir = os.path.abspath(os.path.join(args.input_dir, ".."))
    result_file_dir = os.path.join(up_dir, "best_hparams", "{}".format(args.dataset))
    os.makedirs(result_file_dir, exist_ok=True) # create the directory to save the result
    
    records = reporting.load_records(args.input_dir)
    trial_seeds = records.select("args.trial_seed").unique()
    
    #### testing specifying model selection method
    print("----------------------------- testing specifying model selection method ---------------------------")
    class_name_str = SELECTION_METHODS[0].__name__
    print(class_name_str)
    cls = vars(model_selection)[class_name_str]
    print(cls)
    print("----------------------------- end testing ---------------------------")
    ####
    print("The trial_seeds are: ", trial_seeds)
    assert len(trial_seeds), "The number of trials can not be zero"
    res_dict = {}
    
    
    ''' The output dir structure
    # args.input_dir
      # running record directory 1
        # done
        # err.txt
        # out.txt
        # results.jsonl
      # running record 2
      ...
      # running record n
    # best_hparams
      # dataset1
        # hparams.json, it is the list (w.r.t. model selection methods) of hparam outputs from find_best_params_of_1trial_alg_env_tab
        # alg_env_table.tex / alg_env_table.txt
      # dataset2
      ...
      # datasetm
    '''

    table_name = "alg_env_table.tex" if args.latex else "alg_env_table.txt"
    table_path = os.path.join(result_file_dir, table_name)
    err_path = os.path.join(result_file_dir, "err.txt")
    
    # set the sysetm out stream to be the table file
    sys.stdout = misc.Tee(table_path, "w")
    sys.stderr = misc.Tee(err_path)
    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results for {}}".format(args.dataset))
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    if len(trial_seeds) == 1:
        # print tables and select hyper-parameters for the selection methods
        for selection_method in SELECTION_METHODS:
            if args.latex:
                print()
                print("\\subsection{{Model selection: {}}}".format(
                selection_method.__name__))
            selection_method: model_selection.SelectionMethod
            sm_result = find_best_params_of_1trial_alg_env_tab(args.dataset, selection_method, records, args.latex)
            res_dict[selection_method.__name__] = sm_result
    else:
        raise NotImplementedError
        
    if args.latex:
        print("\\end{document}")
    
    # save the results
    json_str = json.dumps(res_dict, indent=4)
    with open(os.path.join(result_file_dir, "hparams.json"), 'w') as json_file:
        json_file.write(json_str)

    