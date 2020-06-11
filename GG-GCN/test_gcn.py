import sys
import os
sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/gcn')
# sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/../')
import warnings
warnings.filterwarnings('ignore')
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
import argparse
import pathlib
    
# Define model evaluation function
def evaluate(features, support, labels, placeholders, masks=None):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, placeholders, masks)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs_softmax], feed_dict=feed_dict_val)
    return outs_val

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'problem',
        choices=['mis', 'vc', 'ds', 'ca'],
    )

    args = parser.parse_args()
    home = expanduser("~")
    model_dir = f'../trained_models/{args.problem}/GG-GCN'

    # Settings
    feat_dim = 57
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 101, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('diver_num', 1, 'Number of outputs.')
    flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('num_layer', 20, 'number of layers.')
    flags.DEFINE_string('matrix_type', 'dense', 'Model string.')  # 'sparse', 'dense'
    num_supports = 1 + FLAGS.max_degree

    ####### model #######
    model_func = GCN_DEEP_DIVER
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)] 
                    if FLAGS.matrix_type == 'sparse' else [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=(None, feat_dim)) 
                    if FLAGS.matrix_type == 'sparse' else  tf.placeholder(tf.float32, shape=(None, feat_dim)), # featureless: #points
        'labels': tf.placeholder(tf.float32, shape=(None, 2)), # 0: not linked, 1:linked
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    model = model_func(placeholders, input_dim=feat_dim, logging=True)

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

    ####### session #######
    config = tf.ConfigProto()
    tf.device('/cpu:0')
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(model_dir)
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    ####### Train model #######

    ####### data #######

    for data_dir in ["test_small", 'test_medium']:
        data_path = f'../datasets/{args.problem}/{data_dir}'    
        data_files = list(pathlib.Path(data_path).glob('sample_*.pkl'))
        data_files = [str(data_file) for data_file in data_files][:100]
        os.makedirs(f'../ret_model', exist_ok=True)
        logfile = f'../ret_model/{args.problem}_{data_dir}_GG_GCN.txt'
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

            data = read_data(data_files[idd], lp_feat = True)
            ct += 1
            xs, ys, adj, names = data
        
            if FLAGS.matrix_type == 'sparse':
                xs = sparse_to_tuple(sp.lil_matrix(xs))
                support = simple_polynomials(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj_sparse(adj)]
            else:
                support = simple_polynomials_to_dense(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj(adj)]

            # testing step
            outs = evaluate(xs, support, ys, placeholders, None)
            probs =  outs[2]
            
            # calcuate precision recall f1
            y_pred = probs[:,1]
            y_true = np.argmax(ys,axis=1)
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

    