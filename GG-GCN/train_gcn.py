import sys
import os
sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/gcn')
from os.path import expanduser
import time
import scipy.io as sio
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import sklearn.metrics as metrics
from utils import *
from models import GCN_DEEP_DIVER
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import warnings
import pathlib
warnings.simplefilter("ignore")
import gzip
import pickle
import argparse

def log(line, f=None, stdout=True):
    if stdout:
        print(line)
        sys.stdout.flush()
    if f is not None:
        f.write(f'{line}\n')
        f.flush()

def set_train_params():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 101, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('diver_num', 1, 'Number of outputs.')
    flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('num_supports', 2, 'number of supports')
    flags.DEFINE_integer('num_layer', 20, 'number of layers.')
    flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    # dense matrix can enjoey tf parallelism
    # but if the problem have a graph that is too large to fit into memory, we need to use sparse matrix
    flags.DEFINE_string('matrix_type', 'dense', 'Model string.')  # 'sparse', 'dense'
    return FLAGS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'problem',
        choices=['mis', 'vc', 'ds', 'ca'],
    )

    args = parser.parse_args()
    filedir = os.path.dirname(__file__)
    save_model_to = f'{filedir}/../trained_models/{args.problem}/GG-GCN'
    os.makedirs(save_model_to, exist_ok=True)
    
    ####### data #######
    data_path = f"{filedir}/../datasets/{args.problem}/train"    
    data_files = list(pathlib.Path(data_path).glob('sample_*.pkl'))
    data_files = [str(data_file) for data_file in data_files][:500]
    
    if args.problem == 'tsp':
        read_data = read_data_tsp
    elif args.problem == 'vrp':
        read_data = read_data_vrp
    elif args.problem in ['mis', 'ds', 'vc', 'ca']:
        read_data = read_data_general
    elif args.problem == 'sc':
        read_data = read_data_sc
    else:
        raise Exception('unknown problem!')

    ####### model #######
    feat_dim = 57
    FLAGS = set_train_params()
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(FLAGS.num_supports)] 
                    if FLAGS.matrix_type == 'sparse' else [tf.placeholder(tf.float32) for _ in range(FLAGS.num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=(None, feat_dim)) 
                    if FLAGS.matrix_type == 'sparse' else  tf.placeholder(tf.float32, shape=(None, feat_dim)), # featureless: #points
        'labels': tf.placeholder(tf.float32, shape=(None, 2)), # 0: not linked, 1:linked
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    model = GCN_DEEP_DIVER(placeholders, input_dim=feat_dim, logging=True)

    ####### session #######
    config = tf.ConfigProto()
    tf.device('/cpu:0')
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    ####### Train model #######
    log_file=open(f"{save_model_to}/score.txt",'w+')

    samples_per_epoch = len(data_files)
    samples_per_log = samples_per_epoch // 10 
    print(f'dataset size: {len(data_files)}, samples_per_log: {samples_per_log}')
    best_loss = 1e9

    for epoch in range(FLAGS.epochs):
        ct = 0
        t1 = time.time()
        all_loss = []
        all_acc = []
        for idd in range(samples_per_epoch):

            t2 = time.time()
            data = read_data(data_files[idd], lp_feat = True)

            ct += 1
            xs, ys, adj, names = data

            if FLAGS.matrix_type == 'sparse':
                xs = sparse_to_tuple(sp.lil_matrix(xs))
                support = simple_polynomials(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj_sparse(adj)]
            else:
                support = simple_polynomials_to_dense(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj(adj)]
                

            # Construct feed dictionary
            feed_dict = construct_feed_dict(xs, support, ys, placeholders, None)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
            all_loss.append(outs[1])
            all_acc.append(outs[2])
            
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
            saver.save(sess,f"{save_model_to}/model.ckpt")
            best_loss = loss_cur_epoch
        sys.stdout.flush()
    log_file.flush(); log_file.close()
    print("Optimization Finished!")



    

