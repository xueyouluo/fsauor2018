# ======================================== 
# Author: Xueyou Luo 
# Email: xueyou.luo@aidigger.com 
# Copyright: Eigen Tech @ 2018 
# ========================================

'''These codes are copied from eigen-tensorflow'''
import codecs
import csv
import os

import numpy as np
import tensorflow as tf

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 30000

def read_vocab(vocab_file):
    """read vocab from file
    
    Args:
        vocab_file ([type]): path to the vocab file, the vocab file should contains a word each line
    
    Returns:
        list of words
    """

    if not os.path.isfile(vocab_file):
        raise ValueError("%s is not a vaild file"%vocab_file)

    vocab = []
    word2id = {}
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        for i,line in enumerate(f):
            word = line.strip()
            if not word:
                raise ValueError("Got empty word at line %d"%(i+1))
            vocab.append(word)
            word2id[word] = len(word2id)

    print("# vocab size: ",len(vocab))
    return vocab, word2id

def load_embed_file(embed_file):
    """Load embed_file into a python dictionary.

    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:

    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547

    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for i,line in enumerate(f):
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(
                    vec), "All embedding size should be same, but got {0} at line {1}".format(len(vec),i+1)
            else:
                emb_size = len(vec)
    return emb_dict, emb_size

def embedding_dropout(embedding, dropout=0.1):
    vocab_size = tf.shape(embedding)[0]
    mask = tf.nn.dropout(tf.ones([vocab_size]),keep_prob=1-dropout) * (1-dropout)
    mask = tf.expand_dims(mask, 1)
    return mask * embedding

def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size."""
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    else:
        return "/gpu:0"

def _load_pretrained_emb_from_file(name, vocab_file, embed_file, num_trainable_tokens=0, dtype=tf.float32):
    print("# Start to load pretrained embedding...")
    vocab,_ = read_vocab(vocab_file)
    if num_trainable_tokens:
        trainable_tokens = vocab[:num_trainable_tokens]
    else:
        trainable_tokens = vocab
    
    emb_dict, emb_size = load_embed_file(embed_file)
    print("# pretrained embedding size",len(emb_dict),emb_size)

    for token in trainable_tokens:
        if token not in emb_dict:
            if '<average>' in emb_dict:
                emb_dict[token] = emb_dict['<average>']
            else:
                emb_dict[token] = list(np.random.random(emb_size))
    
    emb_mat = np.array([emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
    if num_trainable_tokens:
        emb_mat = tf.constant(emb_mat)
        emb_mat_const = tf.slice(emb_mat,[num_trainable_tokens,0],[-1,-1])
        with tf.device(_get_embed_device(num_trainable_tokens)):
            emb_mat_var = tf.get_variable(name + "_emb_mat_var", [num_trainable_tokens, emb_size])
        return tf.concat([emb_mat_var,emb_mat_const],0,name=name)
    else:
        with tf.device(_get_embed_device(len(vocab))):
            emb_mat_var = tf.get_variable(name,emb_mat.shape,initializer=tf.constant_initializer(emb_mat))
        return emb_mat_var

def create_embedding(name, vocab_size, embed_size, vocab_file=None, embed_file=None, num_trainable_tokens=0, dtype=tf.float32, scope=None):
    '''create a new embedding tensor or load from a pretrained embedding file
    
    Args:
        name: name of the embedding
        vocab_size : vocab size
        embed_size : embeddign size
        vocab_file ([type], optional): Defaults to None. vocab file
        embed_file ([type], optional): Defaults to None. 
        num_trainable_tokens (int, optional): Defaults to 0. the number of tokens to be trained, if 0 then train all the tokens
        dtype ([type], optional): Defaults to tf.float32. [description]
        scope ([type], optional): Defaults to None. [description]
    
    Returns:
        embedding variable
    '''

    with tf.variable_scope(scope or "embedding", dtype=dtype) as scope:
        if vocab_file and embed_file:
            embedding = _load_pretrained_emb_from_file(name, vocab_file, embed_file, num_trainable_tokens, dtype)
        else:
            with tf.device(_get_embed_device(vocab_size)):
                embedding = tf.get_variable(name,[vocab_size,embed_size],dtype)
        return embedding

class DropConnectLayer(tf.layers.Dense):
    def __init__(self, units,
               mode=tf.estimator.ModeKeys.TRAIN,
               keep_prob=0.7,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(DropConnectLayer,self).__init__(  units,
                                                activation=activation,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                activity_regularizer=activity_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint,
                                                trainable=trainable,
                                                name=name,
                                                **kwargs)
        self.mode = mode
        self.keep_prob = keep_prob
        self.mask = None

    def build(self, input_shape):
        from tensorflow.python.layers import base
        from tensorflow.python.framework import tensor_shape
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                            'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                        axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            if self.mask is None:
                mask = tf.ones_like(self.kernel)
                self.mask = tf.nn.dropout(mask, keep_prob=self.keep_prob) * self.keep_prob
            self.kernel = self.kernel * self.mask
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                            shape=[self.units,],
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            dtype=self.dtype,
                                            trainable=True)
        else:
            self.bias = None
        self.built = True
    

class WeightDropLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    '''Apply dropout on hidden-to-hidden weights'''
    
    def __init__(self, num_units, weight_keep_drop=0.7, mode=tf.estimator.ModeKeys.TRAIN, 
                forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):
        """Initialize the parameters for an LSTM cell.
        """
        super(WeightDropLSTMCell,self).__init__( num_units, forget_bias, state_is_tuple, activation, reuse)
        self.w_layer = tf.layers.Dense(4 * num_units)
        self.h_layer = DropConnectLayer(4 * num_units, mode, weight_keep_drop, use_bias=False)

    def build(self, inputs_shape):
        # compatible with tf-1.5
        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`.
            state: An `LSTMStateTuple` of state tensors, each shaped
                `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
                `True`.  Otherwise, a `Tensor` shaped
                `[batch_size x 2 * self.state_size]`.
            Returns:
            A pair containing the new hidden state, and the new state (either a
                `LSTMStateTuple` or a concatenated state, depending on
                `state_is_tuple`).
        """
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        # W * x + b
        inputs = self.w_layer(inputs)
        # U * h(t-1)
        h = self.h_layer(h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=inputs + h, num_or_size_splits=4, axis=1)

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state
