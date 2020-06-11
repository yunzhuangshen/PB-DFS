import  tensorflow as tf
from    tensorflow import keras
from    layers import *
from    metrics import *
from    config import args 

class GCN(keras.Model):

    def __init__(self, output_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.col_feats_dim = 57
        self.row_feats_dim = 26

        # projection layers
        self.col_embedding = Dense(input_dim=self.col_feats_dim,
                                output_dim=args.hidden1,
                                act=tf.nn.relu,
                                wp='col_emb')
        self.row_embedding = Dense(input_dim=self.row_feats_dim,
                                output_dim=args.hidden1,
                                act=tf.nn.relu,
                                wp='row_emb')

        # attention layers
        self.attn_dim = args.hidden1*2+2
        self.attn_vc =  Attention(self.attn_dim, wp='attn_vc')
        self.attn_cv =  Attention(self.attn_dim, wp='attn_cv')
        self.attn_vo =  Attention(self.attn_dim, wp='attn_vo')
        self.attn_co =  Attention(self.attn_dim, wp='attn_co')

        self.attn_dict = {
            'attn_vc': self.attn_vc,
            'attn_cv': self.attn_cv,
            'attn_vo': self.attn_vo,
            'attn_co': self.attn_co,
        } 

        # tri-conv layers
        self.conv = GraphConvolution(self.attn_dict, wp=f'tri_conv_0')

        # 2 fully connected layers at the end of the gcn    
        self.out1 = Dense(input_dim=args.hidden1*2, output_dim=32, act=tf.nn.relu,wp='out1')
        self.out2 = Dense(input_dim=32, output_dim=2, act=lambda x: x, wp='out2')
      

    def call(self, inputs):
        
        col_feats, row_feats, obj_hidden, cv_supp, vc_supp, vo_supp, co_supp, label, mask = inputs
        col_hidden_t0 = self.col_embedding(col_feats)
        row_hidden = self.row_embedding(row_feats)
        input = col_hidden_t0, row_hidden, obj_hidden, cv_supp, vc_supp, vo_supp, co_supp
        output = self.conv(input)
        col_hidden_tn = output[0]
        col_logits = self.out2(self.out1(tf.concat([col_hidden_t0, col_hidden_tn], axis=1)))

        # Weight decay loss
        loss = tf.zeros([])
        for var in self.conv.trainable_variables:
            loss += args.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        loss += masked_softmax_cross_entropy(col_logits, label, mask)
        acc = masked_accuracy(col_logits, label, mask)

        return loss, acc


    def predict(self, inputs):
        col_feats, row_feats, obj_hidden, cv_supp, vc_supp, vo_supp, co_supp, label, mask = inputs
        col_hidden_t0 = self.col_embedding(col_feats)
        row_hidden = self.row_embedding(row_feats)
        input = col_hidden_t0, row_hidden, obj_hidden, cv_supp, vc_supp, vo_supp, co_supp
        output = self.conv(input)
        col_hidden_tn = output[0]
        col_logits = self.out2(self.out1(tf.concat([col_hidden_t0, col_hidden_tn], axis=1)))
        return tf.nn.softmax(col_logits)