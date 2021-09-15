import os
import sys
import os
sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/../')

import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip
from os.path import expanduser
import pyscipopt as scip
import utils


def read_sol_file(filepath):
    sol = set()
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            var_name = line.strip()
            sol.add(var_name)
            sol.add(f't_x{var_name}')
    return sol
    


class SamplingAgent(scip.Branchrule):

    def __init__(self, sol, write_to):
        self.sol = sol
        self.write_to = write_to


    def branchexeclp(self, allowaddcons):

        if self.model.getNNodes() == 1:
            ys = []
            cands = []
            cands_dict = self.model.getMapping()
            name_index_mapping = {}

            for lp_col_idx, name in cands_dict.items():
                cands.append(lp_col_idx)
                if name in self.sol:
                    ys.append(lp_col_idx)
                name_index_mapping[name] = lp_col_idx

            state_ding, obj_coef_idx = utils.extract_ding_variable_features(self.model)
            data = [state_ding, ys, cands, obj_coef_idx]
                
            with gzip.open(self.write_to, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'mapping': name_index_mapping,
                    }, f)
            print(f'write {self.write_to}\n')
            # end the scip solving process
            self.model.interruptSolve()
        else:
            self.model.interruptSolve()
            
        # result = self.model.executeBranchRule('relpscost', False)
        return {"result": scip.SCIP_RESULT.DIDNOTRUN}

def collect_samples(data_dir):
    require_sol = 'eval' not in data_dir and 'time' not in data_dir

    def collect_single(id, sol_path=None):
        sample_file = os.path.join(data_dir, f'sample_{id}.pkl')
        if os.path.exists(sample_file):
            print(f"skipping {sample_file}")
            sys.stdout.flush()
            return 

        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(os.path.join(data_dir, f'{id}.lp'))

        utils.init_scip_params(m, presolving=False, seed=0)

        print(f"begin collect {os.path.join(data_dir, f'sample_{id}.pkl')}")
        sys.stdout.flush()
        branchrule = SamplingAgent(
            sol = read_sol_file(sol_path) if require_sol else set(),
            write_to=os.path.join(data_dir, f'sample_{id}.pkl'))

        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)
        m.optimize()
        m.freeProb()

    if require_sol:         # construct training and test data, must have .sol for each .lp
        ids = []; sols = []
        for i in range(0, 3000):
            sol_filepath = os.path.join(data_dir, f'{i}.sol')
            if os.path.exists(sol_filepath):
                ids.append(i)
                sols.append(sol_filepath)

        for cur_id, cur_sol_path in zip(ids, sols):
            collect_single(cur_id, cur_sol_path)
    else:                   # construct training and test data, for all .lp files
        for i in range(0, 30):
            collect_single(i, None)

def remove_broken_sample(data_dir):

    for i in range(0, 3000):
        sample_file = os.path.join(data_dir, f'sample_{i}.pkl')
        if os.path.exists(sample_file):
            try:
                with gzip.open(sample_file, 'rb') as f:
                    sample = pickle.load(f)
            except:
                print(f'<{sample_file}> is broken. removing...')
                os.remove(sample_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    home = expanduser("~")
    data_dir = f'{home}/storage1/instances/mis/time_1000'
    remove_broken_sample(data_dir)
    collect_samples(data_dir)
    data_dir = f'{home}/storage1/instances/mis/time_3000'
    remove_broken_sample(data_dir)
    collect_samples(data_dir)
    data_dir = f'{home}/storage1/instances/mis/time_5000'
    remove_broken_sample(data_dir)
    collect_samples(data_dir)
    data_dir = f'{home}/storage1/instances/mis/time_7000'
    remove_broken_sample(data_dir)
    collect_samples(data_dir)
    data_dir = f'{home}/storage1/instances/mis/time_9000'
    remove_broken_sample(data_dir)
    collect_samples(data_dir)