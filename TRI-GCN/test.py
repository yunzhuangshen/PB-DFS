import  time, sys, os
from    utils import *
from    models import GCN
from    config import args
import scipy.io as sio
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import sklearn.metrics as metrics
from os.path import expanduser
import gzip
import pickle
import pathlib
import  tensorflow as tf;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('tf version:', tf.__version__)
assert tf.__version__.startswith('2.')


if __name__ == '__main__':

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
        'mis': ['test_1000', 'test_3000'], 
        'sc': ['test_1125_825', 'test_1500_1100'],
        'vc': ['test_1000', 'test_2000'],
        'ds': ['test_1000', 'test_2000'],
        'ca': ['test_150-750', 'test_200-1000']
    }
    
    home = expanduser("~")
    model_dir = os.path.join(home, f'storage/trained_models/{args.problem}/{train_dirs[args.problem]}/gcnlp')
    model = GCN( output_dim=2)
    model.load_weights(f'{model_dir}/model.ckpt')

    ####### data #######

    for data_dir in data_dirs[args.problem]:
        data_path = os.path.join(home, f"storage1/instances/{args.problem}/{data_dir}")    
        data_files = list(pathlib.Path(data_path).glob('sample_*.pkl'))
        data_files = [str(data_file) for data_file in data_files][:100]
        logfile = os.path.join(model_dir, data_dir) + '.log'
        nsamples = len(data_files)
        
        log(f'test dataset: <{data_path}>, number of instances: {nsamples}', logfile)
        log(f'log write to: <{logfile}>', logfile)
        t1 = time.time()
        ct=0

        yss = []
        yhss = []
        ncands = []

        for idd in range(nsamples):
            ct+=1
            data = read_data(data_files[idd])
            col_state, row_state, cv, vo, co, ys = read_data(data_files[idd])
            obj_state = np.zeros((1, args.hidden1), dtype=np.float32)
            mask = np.ones((col_state.shape[0], ), dtype=int)
            ncols = col_state.shape[0]; nrows = row_state.shape[0]
            supp_cv = np.reshape( np.stack(preprocess_adj_lp_dense(cv)), (-1, nrows, ncols))
            supp_vc = np.reshape( np.stack(preprocess_adj_lp_dense(cv.T)), (-1, ncols, nrows))
            supp_vo = np.reshape( np.stack(preprocess_adj_lp_dense(vo)), (-1, 2))
            supp_co = np.reshape( np.stack(preprocess_adj_lp_dense(co)), (-1, 2))

            input = [col_state, row_state, obj_state, supp_cv, supp_vc, supp_vo, supp_co, ys, mask]
            probs = model.predict(input)


            # calcuate precision recall f1
            y_pred = probs[:,1]
            y_true = ys[:,1]
            ncand = len(y_true)            
            yss.append(y_true); yhss.append(y_pred); ncands.append(ncand)


        t2 = time.time()
        log(f'time per instance: {(t2-t1)/nsamples}', logfile=logfile)
        yss = np.concatenate(yss, axis=None)
        yhss = np.concatenate(yhss, axis=None)
        ncands = np.concatenate(ncands, axis=None)
        
        line, stats = calc_classification_metrics(yss, yhss, ncands)
        log(line, logfile)

        line, stats = calc_classification_metrics_top(yss, yhss, ncands)
        log(line, logfile)

        sys.stdout.flush()

    