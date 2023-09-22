# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

cwd = os.getcwd()
sys.path.append(cwd)

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    # the records variable now contains the results of *different steps* (e.g. 0, 300, 600, ..., not only the best/last step)
    ######## my multistep change
    # grouped_records = reporting.get_grouped_records(records).map(lambda group:
    #     { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    # ).filter(lambda g: g["sweep_acc"] is not None) # calculate the sweep_acc and filter out those with no sweep_acc #### the grouped_records also contains the results of *different steps* (e.g. 0, 300, 600, ..., not only the best/last step) #### This step gives the selection_method a list of records from the same (trial_seed, dataset, algorithm, test_env), (I think the only differences of these records are the hyper-parameters and maybe the test_envs, for example, a has test_envs=[1, 2] and b has test_envs=[1], for (trial_seed, dataset, algorithm, 1), both a and b will be choosed, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!this is important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!) and the selection_method just choose the parameters with the largest accuracy(which accuracy to use is according to the selection_method)

    #### begin
    ggr = reporting.get_grouped_records(records) # group according to (trial_seed, dataset, algorithm, *a* test_env), can be choosed if they have some common test_env from test_envs, where *a* test_env means that a running record is choosed if the "test_env" is in the record's test_envs, but not necessaryly be equal.
    ggr_sweep_acc = ggr.map(lambda group:{**group, "sweep_acc": selection_method.sweep_acc(group["records"])}) # the sweep_acc is the test accuracy of the selected running record in the "group["records"]"
    ggr_sweep_acc_filterd = ggr_sweep_acc.filter(lambda g: g["sweep_acc"] is not None)
    grouped_records = ggr_sweep_acc_filterd
    ######## my multistep change ------ end

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS]) # I think this does not change the alg_names, and here I know why defining tuple with one element must add a comma after the element, because it is used to distinguish the declaration of the tuple and other operations (just like the addition operation here)

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names] # collect the used datasets

    ## create a table for algorithms and datasets
    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names] # the table, table[i][j] corresponds to the i-th algorithm and the j-th environment, the (-1)-the enviroment is the average
        for i, algorithm in enumerate(alg_names): # each of this loop is used to handle an algorithm "algorithm[i]"
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc")) # first filter out the records with the same dataset, algorithm, test_env tuple as (dataset, algorithm, test_env) (this will filter out the runnings with the same dataset, algorithm, test_env but with different trials, hparams and so on??), and then select out the sweep_acc of the filtered results
                #### trial_accs is the accuracy of "algorithm" on "dataset" using "test_env" as the test environments of many trials, the size is [trial number]
                mean, err, table[i][j] = format_mean(trial_accs, latex) # calculate the mean and variances of the accuracy in different environments
                means.append(mean)
            if None in means:
                table[i][-1] = "X" # if exist None, then we can not have an average
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))
        #### table[i][j] is the accuracy (+- std) (and it is a string used in latex but not the values) for algorithm i and environment j, and table[i][-1] is the average accurary over the test environments for algorithm i
        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ] # the column labels used in the table
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex) # print out the corresponding table

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names] # table[i][j] is the accuracy for algorithm[i] on dataset[j], and table[i][-1] is the average accuracy on all the datasets for algorithm[i]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names): # algorithm[i] and dataset[j]
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset)) # filter by the (algorithm, dataset) pair
                .group("trial_seed") # group them by the trial seed, [e1, ..] where ei = (trial seed, grouped_records_elements with "trial seed" trial seed)
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean() # get the mean of sweep_acc from all possible test environments at different trial_seeds
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    
    # args = parser.parse_args()
    args = parser.parse_args(["--input_dir", "./domainbed/misc/test_sweep_data"])

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        model_selection.LeaveOneOutSelectionMethod,
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
