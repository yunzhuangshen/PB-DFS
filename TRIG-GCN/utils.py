import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, eigs
import sys
import datetime
import scipy.io as sio
import sklearn.metrics as sk_metrics
import gzip
import math

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # features = features/features.shape[1]
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = sp.coo_matrix(adj)
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_lp_sparse(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = sp.coo_matrix(adj)
    t_k = [adj, adj_normalized]
    return sparse_to_tuple(t_k)

def preprocess_adj_lp_dense(adj):
    def normalize(input):
        input_min = np.min(input, axis=0, keepdims=True)
        input_max = np.max(input, axis=0, keepdims=True)
        input_delta = input_max - input_min
        input_delta[input_delta==0] = 1
        input = (input - input_min)/input_delta
        return input

    t_k = [adj, normalize(adj)]

    return t_k


def simple_polynomials(adj, k):
    """Calculate polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating polynomials up to order {}...".format(k))
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    for i in range(2, k+1):
        t_new = t_k[-1]*laplacian
        t_k.append(t_new)
    
    return sparse_to_tuple(t_k)

def simple_polynomials_to_dense(adj, k):
    """Calculate polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    for i in range(2, k+1):
        t_new = t_k[-1]*laplacian
        t_k.append(t_new)
    
    for i in range(len(t_k)):
        t_k[i] = t_k[i].toarray()

    return t_k



def log(line, logfile=None):        
    line = f'[{datetime.datetime.now()}] {line}' if line is not None else "\n\n\n\n"
    print(line)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(line, file=f)
    sys.stdout.flush()

def read_data(lp_path):
    def normalize(input):
        input_min = np.min(input, axis=0, keepdims=True)
        input_max = np.max(input, axis=0, keepdims=True)
        input_delta = input_max - input_min
        input_delta[input_delta==0] = 1
        input = (input - input_min)/input_delta
        input[np.isnan(input)] = 0
        return input

    with gzip.open(lp_path, 'rb') as f:
        sample = pickle.load(f)
        state, label_cols, cand_cols, _ = sample['data']
        col_state, row_state, cv, vo, co = state
    
    label_cols = np.squeeze(label_cols)
    cand_cols = np.squeeze(cand_cols)
    ys = np.expand_dims(np.isin(cand_cols, label_cols), axis=1)
    ys = np.concatenate([1-ys, ys],axis=1)
    col_state = normalize(np.nan_to_num(col_state[cand_cols])).astype(np.float32)
    row_state = normalize(np.nan_to_num(row_state)).astype(np.float32)
    cv = np.take(cv, cand_cols, axis=1)
    vo = vo[cand_cols]

    return col_state, row_state, cv, vo, co, ys



def calc_classification_metrics_top(y_true, y_pred, ncands=None, top_percentages=[0.80, 0.90 ,0.95, 1], threshold=0.5):
    def calc_single(y_true_single, y_pred_single):
        test_yhats_roundings = (y_pred_single > threshold).astype(int)
        acc = np.sum(y_true_single == test_yhats_roundings) / len(y_true_single)
        precision, recall, f1_score, _ = sk_metrics.precision_recall_fscore_support(
            y_true_single, test_yhats_roundings, labels=[0,1])

        return acc, f1_score[0], f1_score[1], precision[0], precision[1]

    if ncands is None:
        ret = {}
        y_pred_confidence = np.where( y_pred < threshold, y_pred, 1 - y_pred)
        sortedargs = np.argsort(y_pred_confidence)
        for cur_percentage in top_percentages:
            num_vars_choosen = int(len(y_true)*cur_percentage) 
            sortedargs_cur_percentage = sortedargs[:num_vars_choosen]
            stats = calc_single(y_true[sortedargs_cur_percentage], y_pred[sortedargs_cur_percentage])
            ret[cur_percentage] = [[stats[i]] for i in range(len(stats))]
    else:
        ncands = np.insert(ncands, 0,0)
        slices = np.cumsum(ncands)
        mean_stats = {key: [] for key in top_percentages}
        for i in range(len(slices)-1):
            begin = slices[i]; end = slices[i+1]
            y_true_sinlge = y_true[begin:end]; y_pred_single = y_pred[begin:end]
            y_pred_confidence = np.where( y_pred_single < threshold, y_pred_single, 1 - y_pred_single)
            sortedargs = np.argsort(y_pred_confidence)
            for cur_percentage in top_percentages:
                num_vars_choosen = int(len(y_true_sinlge)*cur_percentage) 
                # print(num_vars_choosen, len(y_true))
                sortedargs_cur_percentage = sortedargs[:num_vars_choosen]
                # print(len(sortedargs_cur_percentage), len(sortedargs))
                stats = calc_single(y_true_sinlge[sortedargs_cur_percentage], y_pred_single[sortedargs_cur_percentage])
                mean_stats[cur_percentage].append(stats) 

        ret = {key: [] for key in top_percentages}
        for cur_percentage in top_percentages:
            for stat in zip(*mean_stats[cur_percentage]):
                ret[cur_percentage].append((np.mean(stat) * 100, np.std(stat) * 100))
    line = ""
    for p in top_percentages:
        line += f'percentage vars: {p} mean - acc: {ret[p][0][0]:0.2f}, f1_0: {ret[p][1][0]:0.2f}, f1_1: {ret[p][2][0]:0.2f}, p_0: {ret[p][3][0]:0.2f}, p1: {ret[p][4][0]:0.2f}\n'
        line += f'percentage vars: {p} std - acc: {ret[p][0][1]:0.2f}, f1_0: {ret[p][1][1]:0.2f}, f1_1: {ret[p][2][1]:0.2f}, p_0: {ret[p][3][1]:0.2f}, p1: {ret[p][4][1]:0.2f}\n\n'
    return line, ret


def calc_classification_metrics(y_true, y_pred, ncands=None, threshold=0.5):

    def calc_single(y_true_single, y_pred_single):
        test_yhats_roundings = (y_pred_single > threshold).astype(int)
        acc = np.sum(y_true_single == test_yhats_roundings) / len(y_true_single)
        precision, recall, f1_score, _ = sk_metrics.precision_recall_fscore_support(
            y_true_single, test_yhats_roundings, labels=[0,1])
        avg_precision = sk_metrics.average_precision_score(y_true_single, y_pred_single)

        return acc,  np.nan_to_num(avg_precision), precision[0], recall[0], f1_score[0], precision[1], recall[1], f1_score[1]

    if ncands is None:
        mean_stats = calc_single(y_true, y_pred)
        mean_stats = [[mean_stats[i]] for i in range(len(mean_stats))]
    else:
        ncands = np.insert(ncands, 0,0)
        slices = np.cumsum(ncands)
        metricss = []; APs = []
        for i in range(len(slices)-1):
            begin = slices[i]; end = slices[i+1]
            metrics = calc_single(y_true[begin:end], y_pred[begin:end])
            metricss.append(metrics)
            APs.append(str(float(metrics[1])))
        mean_stats = []
        for stat in zip(*metricss):
            mean_stats.append(np.mean(stat) * 100)
    line = f'acc: {mean_stats[0]:0.2f}, ap: {mean_stats[1]:0.2f}\
            \np_0: {mean_stats[2]:0.2f}, r_0: {mean_stats[3]:0.2f}, f1_0: {mean_stats[4]:0.2f},\
            \np_1: {mean_stats[5]:0.2f}, r_1: {mean_stats[6]:0.2f}, f1_1: {mean_stats[7]:0.2f}'
    line += f"\n APs: {','.join(APs)}"
    return line, mean_stats