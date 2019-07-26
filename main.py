# ======================================== 
# Author: Xueyou Luo 
# Email: xueyou.luo@aidigger.com 
# Copyright: Eigen Tech @ 2018 
# ========================================

import argparse
import json
import time

import numpy as np
import tensorflow as tf

from dataset import DataSet
from model import Model,MLMModel
from utils import *
from conlleval import return_report
from create_pretraining_data import PretrainDataset
from thrid_utils import init_from_checkpoint

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # mode
    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")
    parser.add_argument("--pretrain", type='bool', nargs="?", const=True, default=False, help="Whether to pretrain mlm")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="init_checkpoint file")

    # data
    parser.add_argument("--data_files", type=str, nargs='+', default=None, help="data file for train or inference")
    parser.add_argument("--eval_files", type=str, nargs='+', default=None, help="eval data file for evaluation")
    parser.add_argument("--label_file", type=str, default=None, help="label file")
    parser.add_argument("--vocab_file", type=str, default=None, help="vocab file")
    parser.add_argument("--embed_file", type=str, default=None, help="embedding file to restore")
    parser.add_argument("--out_file", type=str, default=None, help="output file for inference")
    parser.add_argument("--max_len", type=int, default=100, help='max length for doc')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--prob", type='bool', nargs="?", const=True, default=False, help="Whether to export prob")

    # model
    parser.add_argument("--add_transform", type='bool', nargs="?", const=True, default=False, help="Whether to add transform layer")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--init_transition_path", type=str, default=None, help = 'init_transition_path')
    parser.add_argument("--decay_schema", type=str, default='hand', help = 'learning rate decay: exp | hand')
    parser.add_argument("--encoder", type=str, default='gnmt', help="gnmt | elmo")
    parser.add_argument("--decay_steps", type=int, default=10000, help="decay steps")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. RMS: 0.001 | 0.0001")
    parser.add_argument("--focal_loss", type=float, default=2., help="gamma of focal loss")
    parser.add_argument("--embedding_dropout", type=float, default=0.1, help="embedding_dropout")
    parser.add_argument("--max_gradient_norm", type=float, default=1.0, help="Clip gradients to this norm.")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="drop out keep ratio for training")
    parser.add_argument("--weight_keep_drop", type=float, default=0.8, help="weight keep drop")
    parser.add_argument("--l2_loss_ratio", type=float, default=0.0, help="l2 loss ratio")
    parser.add_argument("--rnn_cell_name", type=str, default='lstm', help = 'rnn cell name')
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding_size")
    parser.add_argument("--num_units", type=int, default=300, help="num_units")
    parser.add_argument("--double_decoder", type='bool', nargs="?", const=True, default=False, help="Whether to double decoder size")
    parser.add_argument("--variational_dropout", type='bool', nargs="?", const=True, default=True, help="Whether to use variational_dropout")
    parser.add_argument("--num_trainable_tokens", type=int, default=0, help="num of trainable tokens for embedding")
    parser.add_argument("--ema", type='bool', nargs="?", const=True, default=False, help="Whether to use ema")

    # clf
    parser.add_argument("--target_label_num", type=int, default=4, help="target_label_num")

    # train
    parser.add_argument("--need_early_stop", type='bool', nargs="?", const=True, default=True, help="Whether to early stop")
    parser.add_argument("--patient", type=int, default=5, help="patient of early stop")
    parser.add_argument("--debug", type='bool', nargs="?", const=True, default=False, help="Whether use debug mode")
    parser.add_argument("--num_train_epoch", type=int, default=50, help="training epoches")
    parser.add_argument("--num_train_steps", type=int, default=5000, help="num_train_steps")
    parser.add_argument("--num_warmup_steps", type=int, default=500, help="num_warmup_steps")
    parser.add_argument("--steps_per_stats", type=int, default=20, help="steps to print stats")
    parser.add_argument("--steps_per_save", type=int, default=20, help="steps to print stats")
    parser.add_argument("--steps_per_summary", type=int, default=20, help="steps to save summary")
    parser.add_argument("--steps_per_eval", type=int, default=2000, help="steps to save model")

    parser.add_argument("--checkpoint_dir", type=str, default='/tmp/visual-semantic', help="checkpoint dir to save model")
    

def convert_to_hparams(params):
    hparams = tf.contrib.training.HParams()
    for k,v in params.items():
        hparams.add_hparam(k,v)
    return hparams

def inference(flags):
    print_out("inference data file {0}".format(flags.data_files))
    dataset = DataSet(flags.data_files, flags.vocab_file, flags.label_file, flags.batch_size, max_len=-1, mode='inference')
    hparams = load_hparams(flags.checkpoint_dir,{"mode":'inference','checkpoint_dir':flags.checkpoint_dir+"/best_eval"})
    with tf.Session(config = get_config_proto(log_device_placement=False)) as sess:
        model = Model(hparams)
        model.build()
        
        try:
            model.restore_model(sess)  #restore best solution
        except Exception as e:
            print("unable to restore model with exception",e)
            exit(1)

        scalars = model.scalars.eval(session=sess)
        print("Scalars:", scalars)
        weight = model.weight.eval(session=sess)
        print("Weight:",weight)
        cnt = 0
        predicts = []
        for (source, lengths, _ ) in dataset.get_next(shuffle=False):
            predict = model.inference_clf_one_batch(sess, source, lengths)
            predicts.extend(predict)
            print_out("\r# process {0:.2%}".format(cnt/dataset.data_size),new_line=False)
            cnt += len(predict)
    
        results = []
        for example,pred in zip(dataset._raw_data,predicts):
          features = example.raw_tokens[2:]
          tags = [dataset.i2l[int(t)] for t in pred][2:]
          samples = []
          i = 0
          while i < len(features):
            sample = []
            if tags[i] == 'O':
                sample.append(features[i])
                j = i + 1
                while j < len(features) and tags[j] == 'O':
                    sample.append(features[j])
                    j += 1
                samples.append('_'.join(sample) + '/o')
            else:
                if tags[i][0] != 'B':
                    print(tags[i][0] + ' error start')
                    print(tags)
                    j = i + 1
                else:
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/' + tags[i][-1])
            i = j

          results.append(samples)
    print_out("# Write result to file ...")
    with open(flags.out_file,'w') as f:
        for item in results:
            f.write('  '.join(item) + '\n')
    print_out("# Done")


def eval_ner(results, path, name):
      """
      Run perl script to evaluate model
      """
      if not os.path.exists(path):
          os.mkdir(path)
      output_file = os.path.join(path, name + "_ner_predict.utf8")
      with open(output_file, "w") as f:
          to_write = []
          for block in results:
              for line in block:
                  to_write.append(line + "\n")
              to_write.append("\n")

          f.writelines(to_write)
      eval_lines = return_report(output_file)
      f1 = float(eval_lines[1].strip().split()[-1])
      return eval_lines, f1

def train_eval_clf(model, sess, dataset):
    checkpoint_loss, acc = 0.0, 0.0

    predicts = []
    for i,(source, lengths, targets) in enumerate(dataset.get_next(shuffle=False)):
        batch_loss, accuracy, batch_size, predict = model.eval_clf_one_step(sess, source, lengths, targets)
        predicts.extend(predict)
        checkpoint_loss += batch_loss
        acc += accuracy
        if (i+1) % 100 == 0:
            print_out("# batch %d/%d" %(i+1,dataset.num_batches))

    results = []
    for example,pred in zip(dataset._raw_data,predicts):
        result = []
        for c,g,p in zip(example.raw_tokens,example.labels,pred):
            result.append(" ".join([c,dataset.i2l[int(g)],dataset.i2l[int(p)]]))
        results.append(result)
    eval_lines, f1 = eval_ner(results, model.hparams.checkpoint_dir, 'eval')  
    for line in eval_lines:
        print(line)
    print_out( "# Eval loss %.5f, f1 %.5f" % (checkpoint_loss/i, f1))
    return -1 * f1, checkpoint_loss/i

def train_mlm(flags):
  dataset = PretrainDataset(flags.batch_size, flags.vocab_file, flags.data_files, 
                max_seq_length=flags.max_len, short_seq_prob=0.1, masked_lm_prob=0.15,
                max_predictions_per_seq=20)
  params = vars(flags)
  params['vocab_size'] = len(dataset.tokenizer.w2i)
  hparams = convert_to_hparams(params)
  save_hparams(flags.checkpoint_dir, hparams)
  print(hparams)

  train_graph = tf.Graph()

  with train_graph.as_default():
    train_model = MLMModel(hparams)
    train_model.build()
    initializer = tf.global_variables_initializer()
  
  train_sess = tf.Session(graph=train_graph, config=get_config_proto(log_device_placement=False ))
  train_model.init_model(train_sess, initializer=initializer)
  try:
      train_model.restore_model(train_sess)
  except:
      print_out("unable to restore model, train from scratch")
          
  print_out("# Start to train with learning rate {0}, {1}".format(flags.learning_rate,time.ctime()))

  global_step = train_sess.run(train_model.global_step)
  print("# Global step", global_step)
  for epoch in range(flags.num_train_epoch):
    step_time, checkpoint_loss, acc, iters = 0.0, 0.0, 0.0, 0
    for i,features in enumerate(dataset.get_next()):
        start_time = time.time()
        add_summary = (global_step % flags.steps_per_summary == 0)
        batch_loss, global_step, accuracy,token_num = train_model.train_clf_one_step(train_sess,features, add_summary = add_summary) 
        step_time += (time.time() - start_time)
        checkpoint_loss += batch_loss
        acc += accuracy
        iters += token_num

        if global_step == 0:
            continue

        if (i+1) % flags.steps_per_stats == 0:
            train_acc = (acc / flags.steps_per_stats) * 100
            acc_summary = tf.Summary()
            acc_summary.value.add(tag='accuracy', simple_value = train_acc)
            train_model.summary_writer.add_summary(acc_summary, global_step=global_step)

            print_out(
                "# Epoch %d  global step %d loss %.5f batch %d lr %g "
                "accuracy %.5f wps %.2f step time %.2fs" %
                (epoch+1, global_step, checkpoint_loss/flags.steps_per_stats, i+1, train_model.learning_rate.eval(session=train_sess),
                train_acc, (iters)/step_time, step_time/(flags.steps_per_stats)))
            step_time, checkpoint_loss, iters, acc = 0.0, 0.0, 0, 0.0
        
        if global_step % flags.steps_per_save == 0:
            train_model.save_model(train_sess)

    print_out("# Finsh epoch {1}, global step {0}".format(global_step, epoch+1))

def train_clf(flags):
    dataset = DataSet(flags.data_files, flags.vocab_file, flags.label_file, flags.batch_size, max_len=flags.max_len)
    eval_dataset = DataSet(flags.eval_files, flags.vocab_file, flags.label_file, 5 * flags.batch_size, max_len=flags.max_len)

    params = vars(flags)
    params['vocab_size'] = len(dataset.w2i)
    hparams = convert_to_hparams(params)

    save_hparams(flags.checkpoint_dir, hparams)
    print(hparams)

    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    # with train_graph.as_default():
    train_model = Model(hparams)
    train_model.build()
    tvars = tf.trainable_variables()

    with eval_graph.as_default():
        eval_hparams = load_hparams(flags.checkpoint_dir,{"mode":'eval','checkpoint_dir':flags.checkpoint_dir+"/best_eval"})
        eval_model = Model(eval_hparams)
        eval_model.build()

    train_sess = tf.Session(config=get_config_proto(log_device_placement=False))

    if flags.init_checkpoint:
      print("# load from init_checkpoint")
      scaffold_fn = init_from_checkpoint(flags.init_checkpoint, tvars)
    train_sess.run(tf.global_variables_initializer())

    saved_checkpoint = tf.train.latest_checkpoint(flags.checkpoint_dir)
    if saved_checkpoint:
      try:
          train_model.restore_model(train_sess)
      except:
          print_out("unable to restore model, train from scratch")
          
    print_out("# Start to train with learning rate {0}, {1}".format(flags.learning_rate,time.ctime()))

    global_step = train_sess.run(train_model.global_step)
    print("# Global step", global_step)
    train_model.save_model(train_sess)

    eval_ppls = []
    best_eval = 1000000000
    pre_best_checkpoint = None
    final_learn = 3
    for epoch in range(flags.num_train_epoch):
        step_time, checkpoint_loss, acc, iters = 0.0, 0.0, 0.0, 0
        for i,(source, lengths, targets) in enumerate(dataset.get_next()):
            start_time = time.time()
            add_summary = (global_step % flags.steps_per_summary == 0)
            batch_loss, global_step, accuracy, token_num,_ = train_model.train_clf_one_step(train_sess,source, lengths, targets, add_summary = add_summary, run_info= add_summary and flags.debug) 
            step_time += (time.time() - start_time)
            checkpoint_loss += batch_loss
            acc += accuracy
            iters += token_num

            if global_step == 0:
                continue

            if global_step % flags.steps_per_stats == 0:
                train_acc = (acc / flags.steps_per_stats) * 100
                acc_summary = tf.Summary()
                acc_summary.value.add(tag='accuracy', simple_value = train_acc)
                train_model.summary_writer.add_summary(acc_summary, global_step=global_step)

                print_out(
                    "# Epoch %d  global step %d loss %.5f batch %d/%d lr %g "
                    "accuracy %.5f wps %.2f step time %.2fs" %
                    (epoch+1, global_step, checkpoint_loss/flags.steps_per_stats, i+1,dataset.num_batches, train_model.learning_rate.eval(session=train_sess),
                    train_acc, (iters)/step_time, step_time/(flags.steps_per_stats)))
                step_time, checkpoint_loss, iters, acc = 0.0, 0.0, 0, 0.0

            if global_step % flags.steps_per_eval == 0:
                print_out("# global step {0}, eval model at {1}".format(global_step, time.ctime()))
                checkpoint_path = train_model.save_model(train_sess)
                with tf.Session(graph=eval_graph, config=get_config_proto(log_device_placement=False)) as eval_sess:
                    eval_model.init_model(eval_sess)
                    eval_model.restore_ema_model(eval_sess, checkpoint_path)
                    eval_ppl, eval_loss = train_eval_clf(eval_model, eval_sess, eval_dataset)
                    print_out("# current result {0}, previous best result {1}".format(eval_ppl,best_eval))
                    loss_summary = tf.Summary()
                    loss_summary.value.add(tag='eval_loss', simple_value = eval_loss)
                    train_model.summary_writer.add_summary(loss_summary, global_step=global_step)
                    if eval_ppl < best_eval:
                        pre_best_checkpoint = checkpoint_path
                        eval_model.save_model(eval_sess,global_step)
                        best_eval = eval_ppl
                    eval_ppls.append(eval_ppl)
                if flags.need_early_stop:
                    if early_stop(eval_ppls, flags.patient):
                        print_out("# No loss decrease, restore previous best model and set learning rate to half of previous one")
                        current_lr = train_model.learning_rate.eval(session=train_sess)
                        if final_learn > 0:
                            final_learn -= 1
                        else:
                            print_out("# Early stop, exit")
                            exit(0)
                        if flags.num_warmup_steps == 0:
                          train_model.saver.restore(train_sess, pre_best_checkpoint)
                          lr = tf.assign(train_model.learning_rate, current_lr/5)
                          train_sess.run(lr)
                        if final_learn==1:
                            train_model.saver.restore(train_sess, pre_best_checkpoint)
                            dropout = tf.assign(train_model.dropout_keep_prob, 1.0)
                            emd_drop = tf.assign(train_model.embedding_dropout, 0.0)
                            train_sess.run([dropout,emd_drop])
                        eval_ppls = [best_eval]
                        continue

        print_out("# Finsh epoch {1}, global step {0}".format(global_step, epoch+1))
    print_out("# Best accuracy {0}".format(best_eval))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    if flags.pretrain:
        train_mlm(flags)
    elif flags.mode == 'train':
        train_clf(flags)
    elif flags.mode == 'inference':
        inference(flags)
