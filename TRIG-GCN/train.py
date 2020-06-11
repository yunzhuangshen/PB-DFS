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
from    tensorflow.keras import optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('tf version:', tf.__version__)
assert tf.__version__.startswith('2.')

def log(line, f=None, stdout=True):
    if stdout:
        print(line)
        sys.stdout.flush()
    if f is not None:
        f.write(f'{line}\n')
        f.flush()



# set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


if __name__ == '__main__':

    filedir = os.path.dirname(__file__)
    save_model_to = f'{filedir}/../trained_models/{args.problem}/TRIG-GCN'
    os.makedirs(save_model_to, exist_ok=True)
    ####### model #######
    col_dim = 57; row_dim = 26;     
    model = GCN( output_dim=2)

    ####### data #######
    data_path = f'{filedir}/../datasets/{args.problem}/train'
    data_files = list(pathlib.Path(data_path).glob('sample_*.pkl'))
    data_files = [str(data_file) for data_file in data_files][:500]


    ####### Train model #######
    log_file=open(f"{save_model_to}/score.txt",'w+')

    samples_per_epoch = len(data_files)
    samples_per_log = samples_per_epoch // 20 
    print(f'dataset size: {len(data_files)}, samples_per_log: {samples_per_log}')
    best_loss = 1e9

    optimizer = optimizers.Adam(lr=1e-2)
    for epoch in range(args.epochs):
        
        ct = 0
        t1 = time.time()
        all_loss = []
        all_acc = []
        for idd in range(samples_per_epoch):
            ct += 1
            t2 = time.time()
            col_state, row_state, cv, vo, co, label = read_data(data_files[idd])
            obj_state = np.zeros((1, args.hidden1), dtype=np.float32)
            mask = np.ones((col_state.shape[0], ), dtype=int)
            ncols = col_state.shape[0]; nrows = row_state.shape[0]
            supp_cv = np.reshape( np.stack(preprocess_adj_lp_dense(cv)), (-1, nrows, ncols))
            supp_vc = np.reshape( np.stack(preprocess_adj_lp_dense(cv.T)), (-1, ncols, nrows))
            supp_vo = np.reshape( np.stack(preprocess_adj_lp_dense(vo)), (-1, 2))
            supp_co = np.reshape( np.stack(preprocess_adj_lp_dense(co)), (-1, 2))

            input = [col_state, row_state, obj_state, supp_cv, supp_vc, supp_vo, supp_co, label, mask]
            with tf.GradientTape() as tape:
                loss, acc = model(input)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            all_loss.append(loss)
            all_acc.append(acc)

            if ct % samples_per_log == 0:
                line = '{} {} loss={:.4f} acc={:.4f} time_sample={:.1f}'.format(
                    epoch + 1, ct, 
                    np.mean(all_loss[-samples_per_log:]), np.mean(all_acc[-samples_per_log:]), 
                    time.time() - t2)
                log(line, log_file)

        loss_cur_epoch = np.mean(all_loss)
        line = '[{} finished!] loss={:.4f} acc={:.4f} time_epoch={:.1f}'.format(
            epoch + 1, loss_cur_epoch, np.mean(all_acc), time.time() - t1)
        log(line, log_file)

        if loss_cur_epoch < best_loss:
            log(f'best model currently, save to {save_model_to}', log_file)
            model.save_weights(f'{save_model_to}/model.ckpt')
            best_loss = loss_cur_epoch

        sys.stdout.flush()
    log_file.flush(); log_file.close()
    print("Optimization Finished!")

