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
# import pyscipopt as scip
import time
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


def construct_feed_dict(features, support, labels, placeholders, masks=None):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    if masks is None:
         masks = np.ones([len(labels)], dtype=np.int32)
    feed_dict.update({placeholders['labels_mask']: masks})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def construct_feed_dict4pred(features, support, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigs(laplacian, 1, which='LR', maxiter=5000)
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)



def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = sp.coo_matrix(adj)
    return sparse_to_tuple(adj_normalized)
    
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
    # line += f"\n APs: {','.join(APs)}"
    return line, mean_stats


# def init_scip_params(model, seed, heuristics=False, presolving=False, separating=False, conflict=True):

#     seed = seed % 2147483648  # SCIP seed range

#     # set up randomization
#     model.setBoolParam('randomization/permutevars', False)
#     model.setIntParam('randomization/permutationseed', seed)
#     model.setIntParam('randomization/randomseedshift', seed)

#     # separation only at root node
#     model.setIntParam('separating/maxrounds', 0)

#     # if asked, disable presolving
#     if not presolving:
#         model.setIntParam('presolving/maxrounds', 0)
#         model.setIntParam('presolving/maxrestarts', 0)

#     # if asked, disable separating (cuts)
#     if not separating:
#         model.setIntParam('separating/maxroundsroot', 0)

#     # if asked, disable conflict analysis (more cuts)
#     if not conflict:
#         model.setBoolParam('conflict/enable', False)

#     # if asked, disable primal heuristics
#     if not heuristics:
#         model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


# def extract_ding_variable_features(model):
#     """
#     Extract features following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

#     Parameters
#     ----------
#     model : pyscipopt.scip.Model
#         The current model.
#     candidates : list of pyscipopt.scip.Variable's
#         A list of variables for which to compute the variable features.
#     root_buffer : dict
#         A buffer to avoid re-extracting redundant root node information (None to deactivate buffering).

#     Returns
#     -------
#     variable_features : 2D np.ndarray
#         The features associated with the candidate variables.
#     """

#     col_state = model.getDingStateCols()
#     col_feature_names = sorted(col_state)
#     for index, name in enumerate(col_feature_names):
#         if name == 'col_coefs':
#             break

#     col_state = np.stack([col_state[feature_name] for feature_name in col_feature_names], axis=1)

#     row_state = model.getDingStateRows()
#     row_feature_names = sorted(row_state)
#     row_state = np.stack([row_state[feature_name] for feature_name in row_feature_names], axis=1)

#     vc, vo, co = model.getDingStateLPgraph()

#     return (col_state, row_state, vc, vo, co), index


def load_samples(filenames, logfile=None):
    x, y, ncands = [], [], []
    total_ncands = 0
    for i, filename in enumerate(filenames):
        # try:
        cand_x, cand_y = load_flat_samples(filename, augment_feats=False, normalize_feats=True)
        # except:
        #     continue
        x.append(cand_x)
        y.append(cand_y)
        ncands.append(cand_x.shape[0])
        total_ncands += ncands[-1]
        
        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(filenames)} files processed ({total_ncands} candidate variables)", logfile)
    
    x = np.concatenate(x)
    y = np.concatenate(y)
    ncands = np.asarray(ncands)

    return x, y, ncands


def load_flat_samples(filename, augment_feats=False, normalize_feats=True):
    with gzip.open(filename, 'rb') as file:
        sample = pickle.load(file)

    ding_state, col_1s, col_cands, obj_idx = sample['data']
    col_state, row_state, cv, vo, co = ding_state

    col_cands = np.array(col_cands)
    col_1s = np.array(col_1s)

    col_state = np.nan_to_num(col_state[col_cands])
    ys = np.isin(col_cands, col_1s) * 1
    # feature preprocessing
    col_state = preprocess_variable_features(col_state, interaction_augmentation=augment_feats, normalization=normalize_feats)
    col_state = np.nan_to_num(col_state)
    return col_state, ys


def preprocess_variable_features(features, interaction_augmentation, normalization):
    """
    Features preprocessing following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    features : 2D np.ndarray
        The candidate variable features to preprocess.
    interaction_augmentation : bool
        Whether to augment features with 2-degree interactions (useful for linear models such as SVMs).
    normalization : bool
        Wether to normalize features in [0, 1] (i.e., query-based normalization).

    Returns
    -------
    variable_features : 2D np.ndarray
        The preprocessed variable features.
    """
    # 2-degree polynomial feature augmentation
    if interaction_augmentation:
        interactions = (
            np.expand_dims(features, axis=-1) * \
            np.expand_dims(features, axis=-2)
        ).reshape((features.shape[0], -1))
        features = np.concatenate([features, interactions], axis=1)

    # query-based normalization in [0, 1]
    if normalization:
        features -= features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        max_val[max_val == 0] = 1
        features /= max_val

    return features

def construct_adj_from_lp(cv):
    adj = np.zeros([cv.shape[1], cv.shape[1]], dtype=float)
    for cons_arr in cv:
        indices = np.nonzero(cons_arr)[0]
        for i in range(len(indices)):
            for j in range(i):

                adj[indices[i],indices[j]] = 1
                adj[indices[j],indices[i]] = 1
    return adj

def read_data_general(lp_path, lp_feat=True):

    with gzip.open(lp_path, 'rb') as f:
        sample = pickle.load(f)
        state_ding, label_cols, cand_cols, obj_idx = sample['data']
        col_state, row_state, cv, vo, co = state_ding
        if lp_feat:
            col_state = np.nan_to_num(col_state)
        else:
            col_state = np.repeat(np.expand_dims(col_state[:, obj_idx], axis=1), 32, axis=1)  

    label_cols = np.squeeze(label_cols)
    cand_cols = np.squeeze(cand_cols)

    # graph structure
    cv = np.take(cv, cand_cols, axis=1)
    adj = construct_adj_from_lp(cv)

    
    # features
    xs = col_state[cand_cols]
    xs_min = np.min(xs, axis=0, keepdims=True)
    xs_max = np.max(xs, axis=0, keepdims=True)
    xs_delta = xs_max - xs_min
    xs = (xs - xs_min)/xs_delta
    xs[np.isnan(xs)] = 1

    # labels
    ys = np.expand_dims(np.isin(cand_cols, label_cols), axis=1)
    ys = np.concatenate([1-ys, ys],axis=1)

    mapping = {val:key[3:] for key, val in sample['mapping'].items()}
    names = [ mapping[cand_col] for cand_col in cand_cols]
    return xs, ys, adj, names



def read_data_tsp(lp_path, lp_feat=True):
    with gzip.open(lp_path, 'rb') as f:
        sample = pickle.load(f)
        state_ding, label_cols, cands, obj_idx = sample['data']
        col_state, row_state, cv, vo, co = state_ding
        cv = np.take(cv, cand_cols, axis=1)
        if lp_feat:
            col_state = np.nan_to_num(col_state)
        else:
            col_state = np.repeat(np.expand_dims(col_state[:, obj_idx], axis=1), 32, axis=1)
        label_cols = [int(label_col) for label_col in label_cols]
        label_cols = set(label_cols)
        name_index_mapping = sample['mapping']

    try:
        ncity = int(lp_path.split('/')[-2].split('_')[1])
    except:
        ncity = len(label_cols)
        
    gdata = sio.loadmat(f'/home/ubuntu/storage1/instances/tsp/dual_graph/{ncity}.dual', appendmat=False)
    orderednames = gdata['names']; orderednames = [name.strip() for name in orderednames]; adj = gdata['adj']; adj = adj + adj.transpose()
    xs = []; ys = []
                
    for name in orderednames:
        col_idx = int(name_index_mapping[f't_x{name}'])
        xs.append(col_state[col_idx])
        ys.append(1 if col_idx in label_cols else 0)
    
    xs = np.array(xs); ys = np.array(ys)
    
    xs_min = np.min(xs, axis=0, keepdims=True)
    xs_max = np.max(xs, axis=0, keepdims=True)
    xs_delta = xs_max - xs_min
    xs_delta[xs_delta==0] = 1
    xs = (xs - xs_min)/xs_delta
    xs[np.isnan(xs)] = 0
    return xs, ys, adj, None

def read_data_sc(lp_path, lp_feat=True):

    tokens = lp_path.split('/')
    g_path = '/'.join(tokens[:-1]) + '/' + tokens[-1][7:-3] + 'sc'
    with gzip.open(lp_path, 'rb') as f:
        sample = pickle.load(f)
        state_ding, label_cols, cands, obj_idx = sample['data']
        col_state, row_state, cv, vo, co = state_ding
        cv = np.take(cv, cand_cols, axis=1)

        if lp_feat:
            col_state = np.nan_to_num(col_state)
        else:
            col_state = np.repeat(np.expand_dims(col_state[:, obj_idx], axis=1), 32, axis=1)

        label_cols = [int(label_col) for label_col in label_cols]
        label_cols = set(label_cols)
        name_index_mapping = sample['mapping']
    gdata = sio.loadmat(g_path, appendmat=False)
    orderednames = gdata['names']; orderednames = [name.strip() for name in orderednames]; 
    adj = gdata['adj2']; adj = adj + adj.transpose()

    xs = []; ys = []
    for name in orderednames:
        trans_name = f't_{name}' if 'x' in name else f't_x{name}'
        if 'y' in trans_name:
            continue
        assert(trans_name in name_index_mapping)
        col_idx = int(name_index_mapping[trans_name])
        xs.append(col_state[col_idx])
        ys.append(1 if col_idx in label_cols else 0)

    xs = np.array(xs); ys = np.array(ys)

    xs_min = np.min(xs, axis=0, keepdims=True)
    xs_max = np.max(xs, axis=0, keepdims=True)
    xs_delta = xs_max - xs_min
    xs_delta[xs_delta==0] = 1
    xs = (xs - xs_min)/xs_delta
    return xs, ys, adj, None



def read_data_vrp(lp_path, lp_feat=True):
    with gzip.open(lp_path, 'rb') as f:
        sample = pickle.load(f)
        state_ding, label_cols, cands, obj_idx = sample['data']
        col_state, row_state, cv, vo, co = state_ding
        cv = np.take(cv, cand_cols, axis=1)
        if lp_feat:
            col_state = np.nan_to_num(col_state)
        else:
            col_state = np.repeat(np.expand_dims(col_state[:, obj_idx], axis=1), 32, axis=1)
        label_cols = [int(label_col) for label_col in label_cols]
        label_cols = set(label_cols)
        name_index_mapping = sample['mapping']

    try:
        ncity = int(lp_path.split('/')[-2].split('_')[1])
    except:
        ncity = int(np.ceil(np.sqrt(len(name_index_mapping))))

    gdata = sio.loadmat(f'/home/ubuntu/storage1/instances/vrp/dual_graph/{ncity}.dual', appendmat=False)
    orderednames = gdata['names']; orderednames = [name.strip() for name in orderednames]; adj = gdata['adj']; adj = adj + adj.transpose()

    assert(ncity == gdata['nnodes'])

    xs = []; ys = []; 
    for name in orderednames:
        col_idx = int(name_index_mapping[name])
        xs.append(col_state[col_idx])
        ys.append(1 if col_idx in label_cols else 0)
    xs = np.array(xs); ys = np.array(ys)
    xs_min = np.min(xs, axis=0, keepdims=True)
    xs_max = np.max(xs, axis=0, keepdims=True)
    xs_delta = xs_max - xs_min
    xs_delta[xs_delta==0] = 1
    xs = (xs - xs_min)/xs_delta

    return xs, ys, adj, None



def read_data_mis0(lp_path, lp_feat=True):

    tokens = lp_path.split('/')
    g_path = '/'.join(tokens[:-1]) + '/' + tokens[-1][7:-3] + 'adj'
    with gzip.open(lp_path, 'rb') as f:
        sample = pickle.load(f)
        state_ding, label_cols, cands, obj_idx = sample['data']
        col_state, _, _, _, _ = state_ding
        if lp_feat:
            col_state = np.nan_to_num(col_state)
        else:
            col_state = np.repeat(np.expand_dims(col_state[:, obj_idx], axis=1), 32, axis=1)
        label_cols = [int(label_col) for label_col in label_cols]
        label_cols = set(label_cols)
        name_index_mapping = sample['mapping']
    gdata = sio.loadmat(g_path, appendmat=False)
    adj = gdata['adj']; adj = adj + adj.transpose()
    orderednames = [str(i) for i in range(1, len(adj)+1)]

    xs = []; ys = []
    for name in orderednames:
        trans_name = f't_x{name}'
        col_idx = int(name_index_mapping[trans_name])
        xs.append(col_state[col_idx])
        ys.append(1 if col_idx in label_cols else 0)
    xs = np.array(xs); ys = np.array(ys)

    xs_min = np.min(xs, axis=0, keepdims=True)
    xs_max = np.max(xs, axis=0, keepdims=True)
    xs_delta = xs_max - xs_min
    xs = (xs - xs_min)/xs_delta
    xs[np.isnan(xs)] = 1

    # labels
    ys = np.concatenate([1-np.expand_dims(ys, axis=1), np.expand_dims(ys, axis=1)],axis=1)

    return xs, ys, adj, None