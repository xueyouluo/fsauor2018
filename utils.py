# ======================================== 
# Author: Xueyou Luo 
# Email: xueyou.luo@aidigger.com 
# Copyright: Eigen Tech @ 2018 
# ========================================
import codecs
import json
import os
import sys

import numpy as np
import tensorflow as tf


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def _reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return tf.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return tf.reverse(input_, axis=[seq_dim])

def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def single_rnn_cell(cell_name, num_units, train_phase=True, keep_prob=0.75, weight_keep_drop=0.65, variational_dropout = False):
    """
    Get a single rnn cell
    """
    cell_name = cell_name.upper()
    if cell_name == "GRU":
        cell = tf.contrib.rnn.GRUCell(num_units)
    elif cell_name == "LSTM":
        cell = tf.contrib.rnn.LSTMCell(num_units)
    elif cell_name == 'block_lstm'.upper():
        cell = tf.contrib.rnn.LSTMBlockCell(num_units)
    elif cell_name == 'WEIGHT_LSTM':
        from thrid_utils import WeightDropLSTMCell
        cell = WeightDropLSTMCell(num_units,weight_keep_drop=weight_keep_drop,mode=tf.estimator.ModeKeys.TRAIN if train_phase and weight_keep_drop<1.0 else tf.estimator.ModeKeys.PREDICT)
    elif cell_name == 'LAYERNORM_LSTM':
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(num_units)

    # dropout wrapper
    if train_phase:
        # TODO: variational_recurrent=True and input_keep_prob < 1 then we need provide input_size
        # But because we use different size in different layers, we will got shape in-compatible error
        # So I just set input_keep_prob to 1.0 when we use variational dropout to avoid this error for now.
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob if not variational_dropout else 1.0,
            output_keep_prob=keep_prob,
            variational_recurrent=variational_dropout,
            dtype=tf.float32)

    return cell

def focal_loss(labels, logits, gamma=2):
    epsilon = 1.e-9
    y_pred = tf.nn.softmax(logits,dim=-1)
    y_pred = y_pred + epsilon # to avoid 0.0 in log
    L = -labels*tf.pow((1-y_pred),gamma)*tf.log(y_pred)
    L = tf.reduce_sum(L)
    batch_size = tf.shape(labels)[0]
    return L / tf.to_float(batch_size)

def get_total_param_num(params, threshold = 1):
    total_parameters = 0
    #iterating over all variables
    for variable in params:  
        local_parameters=1
        shape = variable.get_shape()  #getting shape of a variable
        for i in shape:
            local_parameters*=i.value  #mutiplying dimension values
        if local_parameters >= threshold:
            print("variable {0} with parameter number {1}".format(variable, local_parameters))
        total_parameters+=local_parameters
    print('# total parameter number',total_parameters) 
    return total_parameters

def cal_f1(label_num,predicted,truth):
    results = []
    for i in range(label_num):
        results.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    
    for i, p in enumerate(predicted):
        t = truth[i]
        for j in range(label_num):
            if p[j] == 1:
                if t[j] == 1:
                    results[j]['TP'] += 1
                else:
                    results[j]['FP'] += 1
            else:
                if t[j] == 1:
                    results[j]['FN'] += 1
                else:
                    results[j]['TN'] += 1
    
    precision = [0.0] * label_num
    recall = [0.0] * label_num
    f1 = [0.0] * label_num
    for i in range(label_num):
        if results[i]['TP'] == 0:
            if results[i]['FP']==0 and results[i]['FN']==0:
                precision[i] = 1.0
                recall[i] = 1.0
                f1[i] = 1.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0
                f1[i] = 0.0
        else:
            precision[i] = results[i]['TP'] / (results[i]['TP'] + results[i]['FP'])
            recall[i] = results[i]['TP'] / (results[i]['TP'] + results[i]['FN'])
            f1[i] =  2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    # for i in range(label_num):
    #     print(i,results[i], precision[i], recall[i], f1[i])
    return sum(f1)/label_num, sum(precision)/label_num, sum(recall)/label_num


def load_hparams(out_dir, overidded = None):
    hparams_file = os.path.join(out_dir,"hparams")
    print("loading hparams from %s" % hparams_file)
    hparams_json = json.load(open(hparams_file))
    hparams = tf.contrib.training.HParams()
    for k,v in hparams_json.items():
        hparams.add_hparam(k,v)
    if overidded:
        for k,v in overidded.items():
            if k not in hparams_json:
                hparams.add_hparam(k,v)
            else:
                hparams.set_hparam(k,v)
    return hparams

def save_hparams(out_dir, hparams):
    """Save hparams."""
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    hparams_file = os.path.join(out_dir, "hparams")
    print("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())

def get_config_proto(log_device_placement=True, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0, per_process_gpu_memory_fraction=0.95, allow_growth=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = allow_growth
    config_proto.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto

def early_stop(values, no_decrease=3):
    if len(values) < 2:
        return False
    best_index = np.argmin(values)
    if values[-1] > values[best_index] and (best_index + no_decrease) <= len(values):
        return True
    else:
        return False

def gl_stop(values, alpha=5):
    if len(values) < 2:
        return False
    best = -1 * min(values)
    current = -1 * values[-1]
    if 100 * ( 1 - (current / best) ) > alpha:
        return True
    else:
        return False