import os
import sys
# sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/../')
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
import pathlib
import gzip
import warnings
warnings.filterwarnings("ignore")
from utils import log, load_samples, calc_classification_metrics, calc_classification_metrics_top
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb

def load_data(prob_folder, logfile):
    files = list(pathlib.Path(prob_folder).glob('sample_*.pkl'))[:100]
    xs, ys, cands = load_samples(files, logfile)
    x_shift = xs.mean(axis=0)
    x_scale = xs.std(axis=0)
    x_scale[x_scale == 0] = 1
    xs = (xs - x_shift) / x_scale
    
    return xs, ys, cands


if __name__ == '__main__':

    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='the problem to test',
        choices=['tsp', 'vrp', 'sc', 'mis', 'vc', 'ds', 'ca'],
    )

    parser.add_argument(
        '-m', '--model',
        choices=['lr', 'xgb', 'svmlinear'],
        default='lr'
    )

    args = parser.parse_args()

    train_dirs = {
        'tsp': 'train_50-100', 'vrp': 'train_16-25', 
        'mis': 'train_500-1000', 'sc': 'train_750_550_1.5',
        'vc': 'train_500-1000',
        'ds': 'train_500-1000',
        'ca': 'train_100-500-1.5',
    }

    data_dirs = {
        'tsp': ['test_100', 'test_150'], 
        'vrp': ['test_25', 'test_30'], 
        'mis': ['test_1000','test_3000'], 
        'sc': ['test_1125_825', 'test_1500_1100'],
        'vc': ['test_1000', 'test_2000'],
        'ds': ['test_1000', 'test_2000'],
        'ca': ['test_150-750', 'test_200-1000']
    }
    running_dir = os.path.join(home, f'storage/trained_models/{args.problem}/{train_dirs[args.problem]}/{args.model}')


    for data_dir in data_dirs[args.problem]:
        data_path = os.path.join(home, f"storage1/instances/{args.problem}/{data_dir}")    
        logfile = os.path.join(running_dir, data_dir) + '.log'
            
        if args.model in ['lr', 'xgb']:
            
            # load model
            with open(f"{running_dir}/model.pkl", 'rb') as f:
                model = pickle.load(f)

            # load data
            xss, yss, ncands = load_data(data_path, logfile)
            nsamples = len(ncands)
            log(f"test problem: {args.problem} model: {args.model}", logfile)
            log(f'test dataset: <{data_path}>, number of instances: {nsamples}', logfile)
            log(f'log write to: <{logfile}>', logfile)

            yhss = model.predict_proba(xss)[:,1]
            line, stats = calc_classification_metrics(yss, yhss, ncands)
            log(line, logfile)
            line, stats = calc_classification_metrics_top(yss, yhss, ncands)
            log(line, logfile)
        else:

            os.system(f'cd /home/ubuntu/TSP_CPLEX_SVM/build/ && ./TSP test /home/ubuntu/storage/trained_models/tsp/{train_dirs[args.problem]}/svmlinear/ /home/ubuntu/storage1/instances/tsp/{args.test_dir}/')

            # read label files
            pred_files = list(pathlib.Path(data_path).glob('*.svmlinear'))
            pred_files = sorted([str(file) for file in pred_files], key=lambda input: int(input.split("/")[-1][:-10]))
            sol_files = [file[:-9] + "sol" for file in pred_files]
            yss = []; yhss = []; ncands = []
            for pred_file, sol_file in zip(pred_files, sol_files):
                if not os.path.exists(sol_file):
                    continue
                ys = []; yhs = []; pred_vals = {}
                with open(pred_file, 'r') as f:
                    lines = f.readlines(); 
                    pred_vals = {line.strip().split(' ')[0] : float(line.strip().split(' ')[1]) for line in lines}
                with open(sol_file, 'r') as f:
                    true_pos = f.readlines(); true_pos = set([line.strip() for line in true_pos])
                nnodes = len(true_pos)


                for i in range(1, nnodes+1):
                    for j in range(i+1, nnodes+1):
                        if f'{i}_{j}' in pred_vals:
                            yhs.append(pred_vals[f'{i}_{j}'])
                        else:
                            yhs.append(pred_vals[f'{j}_{i}'])

                        ys.append(1 if f'{i}_{j}' in true_pos or f'{j}_{i}' in true_pos else 0)

                yss.append(ys); yhss.append(yhs); ncands.append(nnodes)

                # remove solution files
                # os.remove(sol_file)

            yss = np.concatenate(yss , axis=None); yhss = np.concatenate(yhss , axis=None); ncands = np.array(ncands)
            line, metrics = calc_classification_metrics(yss, yhss, ncands, threshold=0)
            log(line, logfile)
            line, stats = calc_classification_metrics_top(yss, yhss, ncands, threshold=0)
            log(line, logfile)