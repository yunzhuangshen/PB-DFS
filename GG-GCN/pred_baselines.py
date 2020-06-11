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
from utils import load_flat_samples
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
import time 

def load_data(filepath):
    xs, _ = load_flat_samples(filepath, augment_feats=False, normalize_feats=True)
    x_shift = xs.mean(axis=0)
    x_scale = xs.std(axis=0)
    x_scale[x_scale == 0] = 1
    xs = (xs - x_shift) / x_scale
    return xs


if __name__ == '__main__':

    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='the problem to test',
        choices=['mis', 'ds', 'ca', 'vc'],
    )

    parser.add_argument(
        '-m', '--model',
        choices=['lr', 'xgb'],
    )

    args = parser.parse_args()
    running_dir = f'../trained_models/{args.problem}/{args.model}')

    data_path =f'../datasets/{args.problem}/eval_large'   
    files = list(pathlib.Path(data_path).glob('sample_*.pkl'))

    # load model
    with open(f"{running_dir}/model.pkl", 'rb') as f:
        model = pickle.load(f)
    t1 = time.time()
    for filepath in files:
        # load data
        xss = load_data(filepath)
        yhss = model.predict_proba(xss)[:,1]
        # write probability map to file
        print(filepath)
        with open(str(filepath)[:-3] + 'lr_prob', 'w+') as f:
            print('write to ' + str(filepath)[:-3] + 'lr_prob')
            for idx, prob in enumerate(yhss):
                f.write(f'{idx+1} {prob}\n')
    print(f'average time used: {(time.time() - t1)/len(files)}')
