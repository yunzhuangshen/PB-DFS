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
        choices=['mis', 'vc', 'ds', 'ca'],
    )

    parser.add_argument(
        '-m', '--model',
        choices=['lr', 'xgb'],
    )

    args = parser.parse_args()
    running_dir = f'../trained_models/{args.problem}/{args.model}')


    for data_dir in ['test_small', 'test_medium']:
        data_path = f'../datasets/{args.problem}/{data_dir}'
        os.makedirs(f'../ret_model', exist_ok=True)
        logfile = f'../ret_model/{args.problem}_{data_dir}_{args.model}.txt'

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