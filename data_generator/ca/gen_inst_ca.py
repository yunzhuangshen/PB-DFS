import os, sys
import argparse
import numpy as np
import scipy.sparse
import scipy.io as sio
from itertools import combinations
from os.path import expanduser
from os import path
import re
from functools import cmp_to_key
import random
import gurobipy as gp
from gurobipy import *

def generate_ca(random, filename, n_items=100, n_bids=500, min_value=1, max_value=20,
                       value_deviation=0.5, add_item_prob=0.9, max_n_sub_bids=5,
                       additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                       integers=False, warnings=False):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, add_item_prob, random):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.rand(n_items)

    # item compatibilities
    compats = np.triu(random.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while random.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]

        file.write("maximize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" +{price} x{i+1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        ctr=1
        for item_bids in bids_per_item:
            if item_bids:
                file.write(f"C{ctr}:")
                ctr+=1
                for i in item_bids:
                    file.write(f" +1 x{i+1}")
                file.write(f" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(f" x{i+1}")

def solve_single(lp_path, sol_path, time_limit=200):
    print(f'process lp: {lp_path}')

    model = gp.read(lp_path)
    model.setParam('TimeLimit', time_limit) # set a time limit
    model.setParam('OutputFlag', 0) # disable logging
    model.optimize()
    if model.status != GRB.OPTIMAL:
        print(f'problem is too hard to solve within {time_limit}, skipping!')
        return
    
    print(f'problem is solved with {round(model.runtime, 1)} seconds')
    with open( sol_path, 'w+') as f:
        # f.write('Obj: %f\n' % model.objVal)
        for v in model.getVars():
            if int(v.x) == 1 and v.varName[0] == 'x' :
                f.write(f'{v.varName[1:]}\n')


def solve_ca(lp_path, time_limit=200):
    sol_file_path =  f'{lp_path[:-2]}sol'
    if os.path.exists(sol_file_path):
        print(f'{lp_path} has been processed, skipping!')
        return

    solve_single(lp_path, sol_file_path, time_limit=time_limit)
    sys.stdout.flush()


def gen_ca(data_dir, ninst, nitems, nbids, nitems_upper=None, nbids_upper=None, solve=True):
    
    os.makedirs(data_dir, exist_ok=True)

    for i in range(ninst):
        nitems = nitems if nitems_upper is None else random.randint(nitems, nitems_upper+1)
        nbids = nbids if nbids_upper is None else random.randint(nbids, nbids_upper+1)
        rng = np.random.RandomState(i)
        lp_path = os.path.join(data_dir, f'{i}.lp')
        print(f'generate {lp_path}')
        generate_ca(rng, lp_path, nitems, nbids)
        
        if solve:
            sol_path = solve_ca(lp_path)

if __name__ == '__main__':

    home = expanduser("~")
    # data_dir = os.path.join(home, f'storage1/instances/ca/train_100-500-1.5')
    # gen_ca(data_dir, 500, 100, 500, nitems_upper=150, nbids_upper=750, solve=True)

    data_dir = os.path.join(home, f'storage1/instances/ca/time_100-500')
    gen_ca(data_dir, 30, 100, 500, solve=False)
    data_dir = os.path.join(home, f'storage1/instances/ca/time_200-1000')
    gen_ca(data_dir, 30, 200, 1000, solve=False)
    data_dir = os.path.join(home, f'storage1/instances/ca/time_300-1500')
    gen_ca(data_dir, 30, 300, 1500, solve=False)
    data_dir = os.path.join(home, f'storage1/instances/ca/time_400-2000')
    gen_ca(data_dir, 30, 400, 2000, solve=False)
    data_dir = os.path.join(home, f'storage1/instances/ca/time_500-2500')
    gen_ca(data_dir, 30, 500, 2500, solve=False)
