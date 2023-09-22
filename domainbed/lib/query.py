# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Small query library."""

import collections
import inspect
import json
import types
import unittest
import warnings
import math

import numpy as np


def make_selector_fn(selector):
    """
    If selector is a function, return selector.
    Otherwise, return a function corresponding to the selector string. Examples
    of valid selector strings and the corresponding functions:
        x       lambda obj: obj['x']
        x.y     lambda obj: obj['x']['y']
        x,y     lambda obj: (obj['x'], obj['y'])
    """
    if isinstance(selector, str):
        if ',' in selector:
            parts = selector.split(',')
            part_selectors = [make_selector_fn(part) for part in parts]
            return lambda obj: tuple(sel(obj) for sel in part_selectors)
        elif '.' in selector:
            parts = selector.split('.')
            part_selectors = [make_selector_fn(part) for part in parts]
            def f(obj):
                for sel in part_selectors:
                    obj = sel(obj)
                return obj
            return f
        else:
            key = selector.strip()
            return lambda obj: obj[key]
    elif isinstance(selector, types.FunctionType):
        return selector
    else:
        raise TypeError

def hashable(obj):
    try: # some objects such as dictionary, list, set are not hashable(can not use hash function), so they can not be used as the key of a dictionary, so we should use json.sumps to translate them into a string and then we can use the hash of them to be the key of a dictionary
        hash(obj)
        return obj
    except TypeError:
        return json.dumps({'_':obj}, sort_keys=True) # use sort_keys = True is to ensure the uniqueness of the hash of a dictionary/set/list

class Q(object):
    def __init__(self, list_):
        super(Q, self).__init__()
        self._list = list_

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        return self._list[key]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._list == other._list
        else:
            return self._list == other

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return repr(self._list)

    def _append(self, item):
        """Unsafe, be careful you know what you're doing."""
        self._list.append(item)

    def group(self, selector):
        """
        Group elements by selector and return a list of (group, group_records)
        tuples.
        """
        selector = make_selector_fn(selector)
        groups = {}
        for x in self._list:
            group = selector(x) # the selected element of x, it is used as an index
            group_key = hashable(group) # to make sure the index "group" is hashable, see hashable
            if group_key not in groups:
                groups[group_key] = (group, Q([])) # use group_key as the index to accumulate the records xs with the same value "group". Here the authors use the hash value as the index rather than group itself because 
            groups[group_key][1]._append(x)
        results = [groups[key] for key in sorted(groups.keys())] # get values ((group, Q) pair)  of groups, this just divided the elements in self._list into many groups according to the values specified by the selector
        return Q(results)

    def group_map(self, selector, fn):
        """
        Group elements by selector, apply fn to each group, and return a list
        of the results. First group by selector and the map by fn
        """
        return self.group(selector).map(fn)

    def map(self, fn):
        """
        map self onto fn. If fn takes multiple args, tuple-unpacking
        is applied.
        """
        # use fn to act on the elements in self._list and return the handled Q object
        if len(inspect.signature(fn).parameters) > 1: # the number of paramters in function fn
            return Q([fn(*x) for x in self._list])
        else:
            return Q([fn(x) for x in self._list])
        
    def map_with_pnum(self, fn, pnum):
        r'''
        @functional: The same as map, but do not automatically decide the number of params but manually
        @parameters:
            @fn: The same as method map
            @pnum: The number of parameters
        @return: The same as method map
        '''
        if pnum > 1:
            return Q([fn(*x) for x in self._list])
        else:
            return Q([fn(x) for x in self._list])

    
    def select(self, selector):
        '''
        Select the elements in self._list according to the selector
        '''
        selector = make_selector_fn(selector)
        return Q([selector(x) for x in self._list])

    def min(self):
        return min(self._list)

    def max(self):
        return max(self._list)

    def sum(self):
        return sum(self._list)

    def len(self):
        return len(self._list)

    def mean(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.mean(self._list))

    def std(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.std(self._list))

    def mean_std(self):
        return (self.mean(), self.std())

    def argmax(self, selector): # choose max one according to the "arg specified by the selector"
        selector = make_selector_fn(selector)
        return max(self._list, key=selector)

    # filter out the elements in self._list that does not satisfy conditions in fn
    def filter(self, fn):
        return Q([x for x in self._list if fn(x)])

    def filter_equals(self, selector, value): # filter out the elements without particular value requirements
        """like [x for x in y if x.selector == value]"""
        selector = make_selector_fn(selector)
        return self.filter(lambda r: selector(r) == value)

    def filter_not_none(self):
        return self.filter(lambda r: r is not None)

    def filter_not_nan(self):
        return self.filter(lambda r: not np.isnan(r))

    def flatten(self):
        return Q([y for x in self._list for y in x])

    # eliminate the repeated items
    def unique(self):
        result = []
        result_set = set()
        for x in self._list:
            hashable_x = hashable(x)
            if hashable_x not in result_set:
                result_set.add(hashable_x)
                result.append(x)
        return Q(result)

    def sorted(self, key=None):
        if key is None:
            key = lambda x: x
        def key2(x):
            x = key(x)
            if isinstance(x, (np.floating, float)) and np.isnan(x):
                return float('-inf')
            else:
                return x
        return Q(sorted(self._list, key=key2)) # here, the used "sorted" function is given by python but not the class Q
