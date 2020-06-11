import pickle
import sys
import os
# sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/../')
import argparse
import numpy as np
import pathlib
import shutil
import warnings
warnings.filterwarnings("ignore")
from utils import log, load_samples, calc_classification_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb

def load_data(prob_folder, train_size=500):
    rng = np.random.RandomState(0)
    files = rng.permutation(list(pathlib.Path(prob_folder).glob('sample_*.pkl')))

    def load(begin, to):
        # Data loading
        xs, ys, cands = load_samples(files[begin:to], logfile)
        log(f"  {xs.shape[0]} training samples", logfile)

        # Data normalization
        x_shift = xs.mean(axis=0)
        x_scale = xs.std(axis=0)
        x_scale[x_scale == 0] = 1
        xs = (xs - x_shift) / x_scale
        return xs, ys, cands
    train_x, train_y, _ = load(begin=0, to=train_size)
    return train_x, train_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['mis', 'ca', 'ds', 'vc'],
    )

    parser.add_argument(
        '-m', '--model',
        help='Model to be trained.',
        type=str,
        choices=['lr', 'xgb'],
    )

    args = parser.parse_args()
    home = os.path.expanduser("~")

    filedir = os.path.dirname(__file__)
    data_path =  f'{filedir}/../datasets/{args.problem}/train'
    running_dir = f'{filedir}/../trained_models/{args.problem}/{args.model}'
    os.makedirs(running_dir, exist_ok=True)

    logfile = f"{running_dir}/log.txt"
    if args.model in ['lr', 'xgb']:
        log(f"Logfile for {args.model} model on {args.problem}", logfile)
        log(f"training files from {data_path}")
        log(f"model saves to {running_dir}")

        train_xs, train_ys = load_data(data_path)
        if args.model == 'lr':
            model = LogisticRegression()
        elif args.model == 'xgb':
            model = xgb.XGBClassifier()
        
        ###### train session ######

        log("Starting training", logfile)
        model.fit(train_xs, train_ys)
        with open(f"{running_dir}/model.pkl", "wb") as file:
            pickle.dump(model, file)
        log(f"Done training", logfile)