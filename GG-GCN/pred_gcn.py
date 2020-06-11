from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/gcn')
from os.path import expanduser
home = expanduser("~")
import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
import warnings
warnings.simplefilter("ignore")

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from utils import *
from models import GCN_DEEP_DIVER
import time
import argparse
import pathlib
N_bd = 57

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', 1, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'number of layers.')
flags.DEFINE_string('matrix_type', 'sparse', 'Model string.')  # 'sparse', 'dense'


# Some preprocessing

num_supports = 1 + FLAGS.max_degree
model_func = GCN_DEEP_DIVER

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(-0)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Define model evaluation function
def evaluate(features, support, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return (time.time() - t_test), outs_val[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'problem',
        choices=['mis', 'ds', 'vc', 'ca'],
    )

    args = parser.parse_args()

    feat_dim = 57
    # Define placeholders
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
    # Create model
    model = model_func(placeholders, input_dim=N_bd, logging=True)

    args = parser.parse_args()
    home = expanduser("~")
    model_dir = os.path.join(home, f'../trained_models/{args.problem}/GG-GCN')

    data_path = f'../datasets/{args.problem}/eval_large'
    data_files = [f'{data_path}/sample_{i}.pkl' for i in range(30)]

    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(model_dir)
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,os.path.join(home, ckpt.model_checkpoint_path))

    t1 = time.time()
    ct=0
    for data_file in data_files:
        print('processing data file ' + data_file + '\n')
        data = read_data_general(data_file, lp_feat = (args.feature =='lp'))

        ct += 1
        xs, ys, adj, names, = data
    
        if FLAGS.matrix_type == 'sparse':
            xs = sparse_to_tuple(sp.lil_matrix(xs))
            support = simple_polynomials(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj_sparse(adj)]
        else:
            support = simple_polynomials_to_dense(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj(adj)]

        _, z_out = evaluate(xs, support, placeholders)

        prob_map = z_out[:, 1].tolist()
        assert(len(names) == len(prob_map))
        # write probability map to file
        with open(data_file[:-3] + 'prob', 'w+') as f:
            print('write to ' + data_file[:-3] + 'prob')
            for varname, prob in zip(names, prob_map):
                f.write(f'{varname} {prob}\n')
    print(f'average time used: {(time.time() - t1)/len(data_files)}')
