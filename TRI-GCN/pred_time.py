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
        'tsp': ['eval_200'], 
        'vrp': ['eval_50'], 
        'mis': ['time_1000', 'time_3000', 'time_5000', 'time_7000', 'time_9000'], 
        'sc': ['eval_1500_1100', 'eval_2250_1750'],
        'vc': ['time_1000', 'time_3000', 'time_5000', 'time_7000', 'time_9000'],
        'ds': ['time_1000', 'time_3000', 'time_5000', 'time_7000', 'time_9000'],
        'ca': ['time_100-500', 'time_200-1000', 'time_300-1500', 'time_400-2000', 'time_500-2500']
    }
    
    home = expanduser("~")
    model_dir = os.path.join(home, f'storage/trained_models/{args.problem}/{train_dirs[args.problem]}/gcnlp')
    model = GCN( output_dim=2)
    model.load_weights(f'{model_dir}/model.ckpt')

    ####### data #######
    logfile = os.path.join(model_dir, 'prediction_time.log')
    for data_dir in data_dirs[args.problem]:
        data_path = os.path.join(home, f"storage1/instances/{args.problem}/{data_dir}")    
        data_files = list(pathlib.Path(data_path).glob('sample_*.pkl'))
        data_files = [str(data_file) for data_file in data_files][:3]
        nsamples = len(data_files)
        
        t1 = time.time()
        ct=0

        for idd in range(nsamples):
            print('processing data file ' + data_files[idd] + '\n')
            sys.stdout.flush()
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

        t2 = time.time()
        log(f'time per instance: {(t2-t1)/nsamples}', logfile=logfile)
    
