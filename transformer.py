# coding=utf-8

import math
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf

def gelu(x):
    """
    GELU activation, described in paper "Gaussian Error Linear Units (GELUs)"
    https://arxiv.org/pdf/1606.08415.pdf
    """
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + K.tanh(c * (x + 0.044715 * K.pow(x, 3))))


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention():
    def __init__(self, attn_dropout=0.1):
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):   # mask_k or mask_qk
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout, mode=0):
        self.mode = mode
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], s[2]//n_head])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs, attn

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def __call__(self, x):
        output = self.w_1(x) 
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.norm_layer = LayerNormalization()
    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.norm_layer(Add()([enc_input, output]))
        output = self.pos_ffn_layer(output)
        return output, slf_attn

class DecoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.enc_att_layer  = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.norm_layer1 = LayerNormalization()
        self.norm_layer2 = LayerNormalization()
    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
        if dec_last_state is None: dec_last_state = dec_input
        output, slf_attn = self.self_att_layer(dec_input, dec_last_state, dec_last_state, mask=self_mask)
        x = self.norm_layer1(Add()([dec_input, output]))
        output, enc_attn = self.enc_att_layer(x, enc_output, enc_output, mask=enc_mask)
        x = self.norm_layer2(Add()([x, output]))
        output = self.pos_ffn_layer(x)
        return output, slf_attn, enc_attn

def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
        if pos != 0 else np.zeros(d_emb) 
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

def GetPadMask(q, k):
    '''
    shape: [B, Q, K]
    '''
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    '''
    shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
    '''
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class SelfAttention():
    def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
    def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
        if return_att: atts = []
        if src_seq is not None:
            mask = Lambda(lambda x:K.cast(K.greater(x, 0), 'float32'))(src_seq)
        else:
            mask = None
        x = src_emb     
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x

class Decoder():
    def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
    def __call__(self, tgt_emb, tgt_seq, src_seq, enc_output, return_att=False, active_layers=999):
        x = tgt_emb
        self_pad_mask = Lambda(lambda x:GetPadMask(x, x))(tgt_seq)
        self_sub_mask = Lambda(GetSubMask)(tgt_seq)
        self_mask = Lambda(lambda x:K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x:GetPadMask(x[0], x[1]))([tgt_seq, src_seq])
        if return_att: self_atts, enc_atts = [], []
        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
            if return_att: 
                self_atts.append(self_att)
                enc_atts.append(enc_att)
        return (x, self_atts, enc_atts) if return_att else x



class Transformer:
    def __init__(self, i_tokens_len, o_tokens_len, len_limit, d_model=256, \
              d_inner_hid=512, n_head=4, layers=2, dropout=0.1, \
              share_word_emb=False):
        self.i_tokens_len = i_tokens_len
        self.o_tokens_len = o_tokens_len
        self.len_limit = len_limit
        self.d_model = d_model
        self.decode_model = None
        self.readout_model = None
        self.layers = layers
        d_emb = d_model

        self.src_loc_info = True

        d_k = d_v = d_model // n_head
        assert d_k * n_head == d_model and d_v == d_k

        self.pos_emb = PosEncodingLayer(len_limit, d_emb) if self.src_loc_info else None

        self.emb_dropout = Dropout(dropout)

        self.i_word_emb = Embedding(i_tokens_len, d_emb)
        if share_word_emb: 
            assert i_tokens_len == o_tokens_len
            self.o_word_emb = i_word_emb
        else: self.o_word_emb = Embedding(o_tokens_len, d_emb)

        self.encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
        self.decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)
        self.target_layer = TimeDistributed(Dense(o_tokens_len, use_bias=False))

    def compile(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = Input(shape=(None,), dtype='int32')

        src_seq = src_seq_input
        tgt_seq  = Lambda(lambda x:x[:,:-1])(tgt_seq_input)
        tgt_true = Lambda(lambda x:x[:,1:])(tgt_seq_input)

        src_emb = self.i_word_emb(src_seq)
        tgt_emb = self.o_word_emb(tgt_seq)

        if self.pos_emb: 
            src_emb = add_layer([src_emb, self.pos_emb(src_seq)])
            tgt_emb = add_layer([tgt_emb, self.pos_emb(tgt_seq)])
        src_emb = self.emb_dropout(src_emb)

        enc_output = self.encoder(src_emb, src_seq, active_layers=active_layers)
        dec_output = self.decoder(tgt_emb, tgt_seq, src_seq, enc_output, active_layers=active_layers)   
        final_output = self.target_layer(dec_output)

        def get_loss(y_pred, y_true):
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accu(y_pred, y_true):
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)
                
        loss = get_loss(final_output, tgt_true)
        self.ppl = K.exp(loss)
        self.accu = get_accu(final_output, tgt_true)

        self.model = Model([src_seq_input, tgt_seq_input], final_output)
        self.model.add_loss([loss])
        
        self.model.compile(optimizer, None)
        self.model.add_metric(self.ppl, 'ppl')
        self.model.add_metric(self.accu, 'accu')
    
class PosEncodingLayer:
    def __init__(self, max_len, d_emb):
        self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
                           weights=[GetPosEncodingMatrix(max_len, d_emb)])
    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask
    def __call__(self, seq, pos_input=False):
        x = seq
        if not pos_input: x = Lambda(self.get_pos_seq)(x)
        return self.pos_emb_matrix(x)

class AddPosEncoding:
    def __call__(self, x):
        _, max_len, d_emb = K.int_shape(x)
        pos = GetPosEncodingMatrix(max_len, d_emb)
        x = Lambda(lambda x:x+pos)(x)
        return x


add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
# use this because keras may get wrong shapes with Add()([])


#######################################################################################3
######  Vision Transformer (ViT)

# vit 的 图片分割层
class VitImgPatchLayer(Layer):
    def __init__(self, patch_size, patch_dim, d_model, **kwargs):
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.d_model = d_model
        super(VitImgPatchLayer, self).__init__(**kwargs)

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def build(self, input_shape):
        self.rescale = Lambda(lambda x: x / 255.) # 替代 Rescaling(1./255)
        self.patch_proj = Dense(self.d_model)
        super(VitImgPatchLayer, self).build(input_shape)

    def call(self, x):
        # 处理图片，分割
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, self.d_model)


# vit 的 class 和 pos 编码层
class VitPosEncodingLayer(Layer):
    def __init__(self, num_patches, d_model, **kwargs):
        self.num_patches = num_patches
        self.d_model = d_model
        super(VitPosEncodingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pos_emb = self.add_weight("pos_emb", shape=(1, self.num_patches + 1, self.d_model),
            initializer=RandomNormal(mean=0.0, stddev=0.06), trainable=True)
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.d_model),
            initializer=Zeros(), trainable=True)
        super(VitPosEncodingLayer, self).build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # 位置信息
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_patches+1, self.d_model)


class VisionTransformer:
    def __init__(self, image_size, num_classes, d_model=256, 
              d_inner_hid=512, n_head=4, layers=2, dropout=0.1, 
              channels=3, patch_size=16):
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.image_size = image_size
        self.channels = channels
        self.d_model = d_model
        self.decode_model = None
        self.readout_model = None
        self.layers = layers
        d_emb = d_model

        self.src_loc_info = True

        d_k = d_v = d_model // n_head
        assert d_k * n_head == d_model and d_v == d_k

        self.img_patch = VitImgPatchLayer(patch_size, patch_dim, d_model)
        self.pos_emb = VitPosEncodingLayer(num_patches, d_model) if self.src_loc_info else None
        self.emb_dropout = Dropout(dropout)
        self.encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
        self.decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)

        self.mlp_head_dense = Dense(d_inner_hid, activation=gelu)
        self.mlp_head_dropout = Dropout(dropout)
        self.mlp_head_output = Dense(num_classes)


    def compile(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(self.image_size,self.image_size,self.channels), dtype='float32')

        src_seq = src_seq_input

        src_emb = self.img_patch(src_seq)
        if self.pos_emb: 
            src_emb = self.pos_emb(src_emb)
        src_emb = self.emb_dropout(src_emb)

        # mask 传入 None
        enc_output = self.encoder(src_emb, None, active_layers=active_layers)
        
        # First (class token) is used for classification
        x = Lambda(lambda x:x[:,0])(enc_output)

        x = self.mlp_head_dense(x)
        x = self.mlp_head_dropout(x)
        final_output = self.mlp_head_output(x)

        self.model = Model(src_seq_input, final_output)

        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"],
        )
