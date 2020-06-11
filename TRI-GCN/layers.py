from    inits import *
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers
from    config import args
keras.backend.set_floatx('float32')
import sys, os


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res




class Dense(layers.Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, wp='dense', act=tf.nn.relu, bias=True, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.act = act
        self.bias = bias
        self.wp = wp
        self.vars={}
        self.vars[f'{self.wp}_weights'] = self.add_weight(
                                f'{self.wp}_weights', [input_dim, output_dim])
        if self.bias:
            self.vars[f'{self.wp}_bias'] = self.add_weight(
                                f'{self.wp}_bias', [output_dim])
            

    def call(self, inputs):
        x = inputs

        # transform
        output = dot(x, self.vars[f'{self.wp}_weights'], sparse=False)

        # bias
        if self.bias:
            output += self.vars[f'{self.wp}_bias']
        return self.act(output)


class Attention(layers.Layer):
    def __init__(self, input_dim, act=tf.nn.softmax, wp='attn_weights', **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.vars={}
        self.wp=wp
        self.input_dim = input_dim
        self.act = act
        self.vars[self.wp] = self.add_weight(self.wp, [input_dim, 1])
            
    def call(self, input):
        logits = dot(input, self.vars[self.wp], sparse=False)
        norms = self.act(logits, axis=0)
        return norms


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, attn_dict, act=tf.nn.relu, wp='TRI_GCN', **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.vars={}
        self.wp = wp
        self.args = args
        self.weight_dim = args.hidden1*2
        self.act = act
        self.attn_dict = attn_dict
        self.vars[f'{self.wp}_vc_weights'] = self.add_weight(
                                f'{self.wp}_vc_weights', [self.weight_dim, args.hidden1])
        self.vars[f'{self.wp}_cv_weights'] = self.add_weight(
                                f'{self.wp}_cv_weights', [self.weight_dim, args.hidden1])
        self.vars[f'{self.wp}_co_weights'] = self.add_weight(
                                f'{self.wp}_co_weights', [self.weight_dim, args.hidden1])
        self.vars[f'{self.wp}_oc_weights'] = self.add_weight(
                                f'{self.wp}_oc_weights', [self.weight_dim, args.hidden1])
        self.vars[f'{self.wp}_vo_weights'] = self.add_weight(
                                f'{self.wp}_vo_weights', [self.weight_dim, args.hidden1])
        self.vars[f'{self.wp}_ov_weights'] = self.add_weight(
                                f'{self.wp}_ov_weights', [self.weight_dim, args.hidden1])



    def call(self, inputs):
        zero = tf.constant(0, dtype=tf.float32)
        col_hidden, row_hidden, obj_hidden, cv_supp, vc_supp, vo_supp, co_supp = inputs
        ncols = col_hidden.shape[0]
        nrows = row_hidden.shape[0]

        if args.dropout != 0:
            col_hidden = tf.nn.dropout(col_hidden, 1-args.dropout)
            row_hidden = tf.nn.dropout(row_hidden, 1-args.dropout)
            obj_hidden = tf.nn.dropout(obj_hidden, 1-args.dropout)

        # convolve v -> o
        attn_coefs = self.attn_dict['attn_vo'](tf.concat([col_hidden, vo_supp, tf.broadcast_to(input=obj_hidden, shape=[ncols, args.hidden1])], axis=1))
        # v_o_out: (?, 1).T (?, 64) -> (1, 64)
        v_o_out = dot(tf.transpose(attn_coefs), col_hidden) 
        # obj_hidden: (1, 128) (128, 64) -> (1, 64)
        obj_hidden = self.act(dot( tf.concat([obj_hidden, v_o_out], axis=1), self.vars[f'{self.wp}_vo_weights']))
        # convolve v,o -> c
        row_hidden_next = []
        for i in tf.range(nrows):
            neighbor_cols_mask = tf.not_equal(cv_supp[0, i, :], zero)
            attn_coefs = self.attn_dict['attn_cv'](tf.concat([
                    tf.boolean_mask(col_hidden, neighbor_cols_mask, axis=0), 
                    tf.transpose(tf.boolean_mask(cv_supp[:, i, :], neighbor_cols_mask, axis=1)), 
                    tf.broadcast_to(row_hidden[i], [tf.reduce_sum(tf.cast(neighbor_cols_mask, dtype=tf.float32)), args.hidden1])], axis=1))
            # v_out: (?, 1).T (?, 64) -> (1, 64)
            v_out = dot(tf.transpose(attn_coefs), tf.boolean_mask(col_hidden, neighbor_cols_mask, axis=0))
            # o_c_out: (1, 64))
            o_c_out = self.act( dot( tf.concat([obj_hidden, tf.expand_dims(row_hidden[i], 0)], axis=1), self.vars[f'{self.wp}_oc_weights']))
            row_hidden_next.append(self.act(dot(tf.concat([o_c_out, v_out],  axis=1), self.vars[f'{self.wp}_vc_weights'])))

        row_hidden_next = tf.squeeze(tf.stack(row_hidden_next))

        # convolve c -> o
        attn_coefs = self.attn_dict['attn_co'](tf.concat([row_hidden_next, co_supp, tf.broadcast_to(obj_hidden, [nrows, args.hidden1])], axis=1))
        c_o_out = dot(tf.transpose(attn_coefs), row_hidden_next)
        obj_hidden = self.act(dot(tf.concat([obj_hidden, c_o_out], axis=1), self.vars[f'{self.wp}_co_weights']))


        # convolve c,o -> v
        col_hidden_next = []
        for i in  tf.range(ncols):
            neighbor_rows_mask = tf.not_equal(vc_supp[0,i,:], zero)
            attn_coefs = self.attn_dict['attn_vc'](tf.concat([
                tf.boolean_mask(row_hidden_next, neighbor_rows_mask, axis=0),
                tf.transpose(tf.boolean_mask(vc_supp[:, i, :], neighbor_rows_mask, axis=1)),
                tf.broadcast_to(col_hidden[i], [tf.reduce_sum(tf.cast(neighbor_rows_mask, dtype=tf.float32)), args.hidden1])], axis=1))
            c_out = dot(tf.transpose(attn_coefs), tf.boolean_mask(row_hidden_next, neighbor_rows_mask, axis=0)) 
            o_v_out = self.act( dot( tf.concat([obj_hidden, tf.expand_dims(col_hidden[i], 0)], axis=1), self.vars[f'{self.wp}_ov_weights']))
            col_hidden_next.append(self.act(dot(tf.concat([o_v_out, c_out], axis=1), self.vars[f'{self.wp}_cv_weights'])))
        col_hidden_next = tf.squeeze(tf.stack(col_hidden_next))
        return col_hidden_next, row_hidden_next, obj_hidden, cv_supp, vc_supp, vo_supp, co_supp