# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np

SELECT_METHODS = [
    'OracleSelectionMethod',
    'IIDAccuracySelectionMethod',
    'LeaveOneOutSelectionMethod'
]

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        
        the run_records are the records with the same (trial_seed, dataset, algorithm, *a* test_env) and the same hparams_seed but maybe different test_envs
        
        !!! we can think of the run_records are records from the same hyper-parameters (if their trial_seeds are the same) !!!
        """
        raise NotImplementedError
    
    @classmethod
    def run_acc_args_hparams(self, run_records):
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples. ZX:And the elements(record) in records has the same 'args.hparams_seed'
        """
        ######## my multistep change
        # return (records.group('args.hparams_seed')
        #     .map(lambda _, run_records:
        #         (
        #             self.run_acc(run_records),
        #             run_records
        #         )
        #     ).filter(lambda x: x[0] is not None)
        #     .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        # )
        #### begin
        g = records.group('args.hparams_seed') # split records into different groups according to 'args.hparams_seed', tbe data structure is : (1) [element1, element2, ...] (2) where elements in (1) are (args.hparams_seed, records) (3) where records in (2) are the list of records with *the same args.hparams_seed* as specified by the first term of the tuple.
        #### if the trial_seeds of the records are the samem, then g can be seem as a list of groups, each group represents a chooice of hyper-parameters of form (hparam_seed, records) where the record in records has the same hparam_seed (i.e. hyper-parameters)
        g_map = g.map(lambda _, run_records:(self.run_acc(run_records), run_records)) # map the elements in g to be (run_acc, records), ignore the group id, so use _. The resulting "run_acc" is the {'val_acc': ... , 'test_acc': ...} dictionary of some record selected from "records" by some concrete model selection method
        g_map_filtered = g_map.filter(lambda x: x[0] is not None) # filter out the elements which has a None 'run_acc' term, which has no test_env, so returned None
        g_map_filtered_sorted = g_map_filtered.sorted(key=lambda x: x[0]['val_acc'])[::-1] # sort the filtered results by the 'val_acc' in a decreasing order
        return g_map_filtered_sorted
        ######## my multistep change ------ end
        
    @classmethod
    def hparams_accs_args_hparams(self, records):
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc_args_hparams(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs): # since the _hparams_accs are sorted according to the val_acc, so _hparams_accs[0] has the greatest val_acc "_hparams_accs[0][0]['test_acc']"
            return _hparams_accs[0][0]['test_acc'] # return the 'test_acc' of the best from different 'args.hparams_seed', model selected according to the specific selection method
            #### return the test acc of the group (a choice of hyper-parameters) with largest val_acc
        else:
            return None
        
    @classmethod
    def sweep_acc_args_hparams_1trial(self, records):
        '''
        Given all records from a single (dataset, algorithm, test env) pair,
        return the (run_acc, args, hparams) tuple with the top val acc for 1-trial sweeps,
        but here the run_acc has the 'step' item
        '''
        _hparams_accs = self.hparams_accs_args_hparams(records)
        if len(_hparams_accs): # since the _hparams_accs are sorted according to the val_acc, so _hparams_accs[0] has the greatest val_acc "_hparams_accs[0][0]['test_acc']"
            train_args = [r['args'] for r in _hparams_accs[0][1]]
            args_different = False
            if len(train_args) > 1:
                args_different = any([arg!=train_args[0] for arg in train_args])
            
            # in fact, if args are the same, then the random hparams will be the same because the random hparams just depend on the (algorithm, dataset, trial_seed, hparam_seed) and they are all specified by the args
            hparams = [r['args'] for r in _hparams_accs[0][1]]
            hparams_different = False
            if len(hparams) > 1:
                hparams_different = any([hparam!=hparams[0] for hparam in hparams])
            assert not args_different, "the args are different"
            assert not hparams_different, "the hparams are different"
            return (_hparams_accs[0][0], _hparams_accs[0][1][0]['args'], _hparams_accs[0][1][0]['hparams']) # return the (run_acc the args, hparams) tuple of the best from different 'args.hparams_seed', model selected according to the specific selection method
            #### return the test acc of the group (a choice of hyper-parameters) with largest val_acc
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records): # choose the accuracy of the last step among the many step records run in a script
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1) # choose the records with just one test env, this just does the same thing as the "get_test_records" function
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0] # here we ensured that the records have the same test_env (dated from the function "get_grouped_records", so we just choose the test env of the first one)
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1] # choose the record with largest step !!!! why not choose according to the test_out_acc (i.e. val_acc) ?????
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }
        
    @classmethod
    def run_acc_args_hparams(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1) # choose the records with just one test env, this just does the same thing as the "get_test_records" function
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0] # here we ensured that the records have the same test_env (dated from the function "get_grouped_records", so we just choose the test env of the first one)
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1] # choose the record with largest step !!!! why not choose according to the test_out_acc (i.e. val_acc) ?????
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key],
            'step': chosen_record['step']
        }
        
class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given *a single record*, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0] # in ID acc selection, the record here has only on element in test_envs
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record: # environment iteration finished
                break
            if i != test_env: # do not consider the test env, just accumulate the training env
                val_env_keys.append(f'env{i}_out_acc') # ID acc selection method use the out_acc as the val accuracy
        test_in_acc_key = 'env{}_in_acc'.format(test_env) # test_in_acc
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }
        
    @classmethod
    def _step_acc_args_hparams(self, record):
        """Given *a single record*, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0] # in ID acc selection, the record here has only on element in test_envs
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record: # environment iteration finished
                break
            if i != test_env: # do not consider the test env, just accumulate the training env
                val_env_keys.append(f'env{i}_out_acc') # ID acc selection method use the out_acc as the val accuracy
        test_in_acc_key = 'env{}_in_acc'.format(test_env) # test_in_acc
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key],
            'step': record['step']
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records) # choose the runs with just one test_envs, because ID acc model selection method just needs a test env
        if not len(test_records):
            return None
        ######## my multistep change
        # return test_records.map(self._step_acc).argmax('val_acc') # after mapping the _step_acc, the result becomes [d1, ..., dn] where di is {'val_acc': ... , 'test_acc': ...} of the i-th record in test_records; and after the argmax it becomes a value "acc"
        #### begin
        v_t_accs = test_records.map(self._step_acc)
        max_v_t_acc = v_t_accs.argmax('val_acc')
        return max_v_t_acc
        ######## my multistep change ------ end
        
    @classmethod
    def run_acc_args_hparams(self, run_records):
        test_records = get_test_records(run_records) # choose the runs with just one test_envs, because ID acc model selection method just needs a test env
        if not len(test_records):
            return None
        ######## my multistep change
        # return test_records.map(self._step_acc).argmax('val_acc') # after mapping the _step_acc, the result becomes [d1, ..., dn] where di is {'val_acc': ... , 'test_acc': ...} of the i-th record in test_records; and after the argmax it becomes a value "acc"
        #### begin
        v_t_accs = test_records.map(self._step_acc_args_hparams)
        max_v_t_acc = v_t_accs.argmax('val_acc')
        return max_v_t_acc
        ######## my multistep change ------ end

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a *single step*. the records has the same (trial_seed, dataset, algorithm, *a* test_env, hparams_seed) and the same step"""
        test_records = get_test_records(records) # choose the records with just one test_env, now the remaining records has the same (trial_seed, dataset, algorithm, test_envs, hparams_seed, step)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0] # find the test environment
        n_envs = 0 # the number of envs
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2): # choose to iterate the records with two envs
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop() # the test_env is the common env that all the records in parameter "records" contains in the "test_envs". Use pop to get the val env in the set(it has only one element, which is just the val_env)
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)] # leave one domain out uses the in_acc as the val accuracy
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:]) # remove the item corresponding to the test_env
        if any([v==-1 for v in val_accs]): # all the val env must have a accuracy, otherwise return None
            return None
        val_acc = np.sum(val_accs) / (n_envs-1) # calculate the **average** val acc
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }
        
    @classmethod
    def _step_acc_args_hparams(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a *single step*. the records has the same (trial_seed, dataset, algorithm, *a* test_env, hparams_seed) and the same step"""
        test_records = get_test_records(records) # choose the records with just one test_env, now the remaining records has the same (trial_seed, dataset, algorithm, test_envs, hparams_seed, step)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0] # find the test environment
        n_envs = 0 # the number of envs
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2): # choose to iterate the records with two envs
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop() # the test_env is the common env that all the records in parameter "records" contains in the "test_envs". Use pop to get the val env in the set(it has only one element, which is just the val_env)
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)] # leave one domain out uses the in_acc as the val accuracy
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:]) # remove the item corresponding to the test_env
        if any([v==-1 for v in val_accs]): # all the val env must have a accuracy, otherwise return None
            return None
        val_acc = np.sum(val_accs) / (n_envs-1) # calculate the **average** val acc
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)],
            'step': test_records[0]['step']
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none() # val_test_acc results of different steps
        if len(step_accs):
            return step_accs.argmax('val_acc') # choose the record with largest val_acc according to different steps
        else:
            return None
        
    @classmethod
    def run_acc_args_hparams(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc_args_hparams(step_records)
        ).filter_not_none() # val_test_acc results of different steps
        if len(step_accs):
            return step_accs.argmax('val_acc') # choose the record with largest val_acc according to different steps
        else:
            return None
