import sys,os
import pandas as pd
import argparse
import shutil
import numpy as np
from scipy.stats.mstats import gmean

def analyse(datedir, problem):

    def geo_mean(arr, mask=None):
        if mask is None:
            arr = arr.to_numpy().astype(float)
        else:
            arr = arr.to_numpy()[~mask].astype(float)
        shift=1
        log_a = np.log(arr+shift)
        return np.abs(np.exp(log_a.mean(axis=0)) - shift)

    prob_dir = f'/home/joey/projects/co_heur_ret/{datedir}/{problem}' 
    methods = [sub_dir for sub_dir in os.listdir(prob_dir) if os.path.isdir(os.path.join(prob_dir, sub_dir))]
    method_dir_paths = [os.path.join(prob_dir, sub_dir) for sub_dir in methods]
    dfs = []

    for method_name, method_dir in zip(methods, method_dir_paths):
        print(method_name)
        fnames = os.listdir(method_dir)
        fnames = [os.path.join(method_dir, fname) for fname in fnames if '.csv' in fname]
        fcontents = []
        for fname in fnames:
            with open(fname, 'r') as f:
                fcontents.append(f.readlines())
                fcontents[-1] = [line.strip() for line in fcontents[-1]]

        content = fcontents[0]
        for fcontent in fcontents[1:]:
             content.extend(fcontent[1:])

        for i in range(len(content)):
            content[i] = content[i].split(',')

        dic = {}
        for arr in zip(*content):
            col_name = arr[0]
            col_content = arr[1:]
            dic[col_name] = col_content
        df = pd.DataFrame.from_dict(dic)

        print(f"problem: {problem} method: {method_name}")
        try:
            if method_name in ['ml_dfs1', 'ml_dfs2', 'scip_agg', 'ml_ding', 'scip_def'] or 'exact' in method_name:
                print('opt_gap', geo_mean(df['opt_gap']))
                print('best_sol_obj', geo_mean(df['best_sol_obj']))
                print('best_sol_time', geo_mean(df['best_sol_time']))
                print('heur_tot_time', geo_mean(df['heur_tot_time']))
            else:
                mask = df['best_heur_sol_obj'].to_numpy() == '0'
                print('nproblems not find solution:', np.sum(mask.astype(int)))
                print('best_heur_sol_obj', geo_mean(df['best_heur_sol_obj'], mask))
                print('best_heur_sol_time', geo_mean(df['best_heur_sol_time'], mask))
                print('heur_ncalls', geo_mean(df['heur_ncalls']))
                print('heur_tot_time', geo_mean(df['heur_tot_time']))
        except:
            pass
        print("\n")
if __name__ == '__main__':

    datedir = '05-Jun-2020'
    problems = ['ds']

    for problem in problems:
        analyse(datedir, problem)
