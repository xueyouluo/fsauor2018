# ========================================
# Author: Xueyou Luo
# Email: xueyou.luo@aidigger.com
# Copyright: Eigen Tech @ 2018
# ========================================
import os

import numpy as np
import tensorflow as tf

from utils import (_reverse, focal_loss, gelu, get_total_param_num, print_out,
                   single_rnn_cell)
from thrid_utils import create_embedding, gather_indexes, layer_norm


class Model(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def is_training(self):
        return self.hparams.mode == 'train'

    def build(self):
        self.setup_input_placeholders()
        self.setup_embedding()
        if self.hparams.encoder == 'gnmt':
            self.gnmt_encoder()
        elif self.hparams.encoder == 'elmo':
            self.elmo_encoder()
        else:
            raise ValueError("Un-supported encoder %s" % self.hparams.encoder)
        self.setup_clf()

        self.params = tf.trainable_variables()
        if self.hparams.ema:
          self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        if self.hparams.mode in ['train', 'eval']:
            self.setup_loss()
        if self.hparams.mode == 'train':
            self.setup_training()
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def init_model(self, sess, initializer=None):
        if initializer:
            sess.run(initializer)
        else:
            sess.run(tf.global_variables_initializer())

    def save_model(self, sess, global_step=None):
        return self.saver.save(sess, os.path.join(self.hparams.checkpoint_dir,
                                                  "model.ckpt"), global_step=global_step if global_step else self.global_step)

    def restore_best_model(self, sess):
        self.saver.restore(sess, tf.train.latest_checkpoint(
            self.hparams.checkpoint_dir + '/best_dev'))

    def restore_ema_model(self, sess, path):
        if self.hparams.ema:
          shadow_vars = {self.ema.average_name(v): v for v in self.params}
          saver = tf.train.Saver(shadow_vars)
        else:
          saver = tf.train.Saver(self.params)
        saver.restore(sess, path)

    def restore_model(self, sess, epoch=None):
        if epoch is None:
            self.saver.restore(sess, tf.train.latest_checkpoint(
                self.hparams.checkpoint_dir))
        else:
            self.saver.restore(
                sess, os.path.join(self.hparams.checkpoint_dir, "model.ckpt" + ("-%d" % epoch)))
        print("restored model")

    def setup_input_placeholders(self):
        self.source_tokens = tf.placeholder(
            tf.int32, shape=[None, None], name='source_tokens')

        # for training and evaluation
        if self.hparams.mode in ['train', 'eval']:
            self.target_labels = tf.placeholder(
                tf.int32, shape=[None, None], name='target_labels')

        self.batch_size = tf.shape(self.source_tokens, out_type=tf.int32)[0]

        self.sequence_length = tf.placeholder(
            tf.int32, shape=[None], name='sequence_length')

        self.global_step = tf.train.get_or_create_global_step()

        self.predict_token_num = tf.reduce_sum(self.sequence_length)
        self.embedding_dropout = tf.Variable(
            self.hparams.embedding_dropout, trainable=False)
        self.dropout_keep_prob = tf.Variable(
            self.hparams.dropout_keep_prob, trainable=False)

    def setup_embedding(self):
        # load pretrained embedding
        self.embedding = create_embedding(
            "embedding",
            self.hparams.vocab_size,
            self.hparams.embedding_size,
            vocab_file=self.hparams.vocab_file,
            embed_file=self.hparams.embed_file,
            num_trainable_tokens=self.hparams.num_trainable_tokens)

        if self.hparams.embedding_dropout > 0 and self.is_training():
            vocab_size = tf.shape(self.embedding)[0]
            mask = tf.nn.dropout(tf.ones(
                [vocab_size]), keep_prob=1 - self.embedding_dropout) * (1 - self.embedding_dropout)
            mask = tf.expand_dims(mask, 1)
            self.embedding = mask * self.embedding

        self.source_embedding = tf.nn.embedding_lookup(
            self.embedding, self.source_tokens)

        if self.is_training():
            self.source_embedding = tf.nn.dropout(
                self.source_embedding, keep_prob=self.dropout_keep_prob)

    def elmo_encoder(self):
        print_out("build elmo encoder")
        with tf.variable_scope("elmo_encoder") as scope:
            inputs = tf.transpose(self.source_embedding, [1, 0, 2])
            inputs_reverse = _reverse(
                inputs, seq_lengths=self.sequence_length,
                seq_dim=0, batch_dim=1)
            encoder_states = []
            outputs = [tf.concat([inputs, inputs], axis=-1)]
            fw_cell_inputs = inputs
            bw_cell_inputs = inputs_reverse
            for i in range(self.hparams.num_layers):
                with tf.variable_scope("fw_%d" % i) as s:
                    cell = tf.contrib.rnn.LSTMBlockFusedCell(
                        self.hparams.num_units, use_peephole=False)
                    fused_outputs_op, fused_state_op = cell(
                        fw_cell_inputs, sequence_length=self.sequence_length, dtype=inputs.dtype)
                    encoder_states.append(fused_state_op)
                with tf.variable_scope("bw_%d" % i) as s:
                    bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(
                        self.hparams.num_units, use_peephole=False)
                    bw_fused_outputs_op_reverse, bw_fused_state_op = bw_cell(
                        bw_cell_inputs, sequence_length=self.sequence_length, dtype=inputs.dtype)
                    bw_fused_outputs_op = _reverse(
                        bw_fused_outputs_op_reverse, seq_lengths=self.sequence_length,
                        seq_dim=0, batch_dim=1)
                    encoder_states.append(bw_fused_state_op)
                output = tf.concat(
                    [fused_outputs_op, bw_fused_outputs_op], axis=-1)
                if i > 0:
                    fw_cell_inputs = output + fw_cell_inputs
                    bw_cell_inputs = _reverse(
                        output, seq_lengths=self.sequence_length,
                        seq_dim=0, batch_dim=1) + bw_cell_inputs
                else:
                    fw_cell_inputs = output
                    bw_cell_inputs = _reverse(
                        output, seq_lengths=self.sequence_length,
                        seq_dim=0, batch_dim=1)
                outputs.append(output)

            final_output = None
            # embedding + num_layers
            n = 1 + self.hparams.num_layers
            scalars = tf.get_variable(
                'scalar', initializer=tf.constant([1 / (n)] * n))
            self.scalars = scalars
            weight = tf.get_variable('weight', initializer=tf.constant(0.001))
            self.weight = weight

            soft_scalars = tf.nn.softmax(scalars)
            for i, output in enumerate(outputs):
                if final_output is None:
                    final_output = soft_scalars[i] * \
                        tf.transpose(output, [1, 0, 2])
                else:
                    final_output = final_output + \
                        soft_scalars[i] * tf.transpose(output, [1, 0, 2])

            self.final_outputs = weight * final_output
            self.final_state = tuple(encoder_states)

    def gnmt_encoder(self):
        print_out("build gnmt encoder")
        with tf.variable_scope("gnmt_encoder") as scope:
            inputs = tf.transpose(self.source_embedding, [1, 0, 2])
            inputs_reverse = _reverse(
                inputs, seq_lengths=self.sequence_length,
                seq_dim=0, batch_dim=1)
            encoder_states = []
            outputs = [inputs]

            with tf.variable_scope("fw") as s:
                cell = tf.contrib.rnn.LSTMBlockFusedCell(
                    self.hparams.num_units, use_peephole=False)
                fused_outputs_op, fused_state_op = cell(
                    inputs, sequence_length=self.sequence_length, dtype=inputs.dtype)
                encoder_states.append(fused_state_op)
                outputs.append(fused_outputs_op)

            with tf.variable_scope('bw') as s:
                bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(
                    self.hparams.num_units, use_peephole=False)
                bw_fused_outputs_op, bw_fused_state_op = bw_cell(
                    inputs_reverse, sequence_length=self.sequence_length, dtype=inputs.dtype)
                bw_fused_outputs_op = _reverse(
                    bw_fused_outputs_op, seq_lengths=self.sequence_length,
                    seq_dim=0, batch_dim=1)
                encoder_states.append(bw_fused_state_op)
                outputs.append(bw_fused_outputs_op)

            with tf.variable_scope("uni") as s:
                uni_inputs = tf.concat(
                    [fused_outputs_op, bw_fused_outputs_op], axis=-1)
                for i in range(self.hparams.num_layers - 1):
                    with tf.variable_scope("layer_%d" % i) as scope:
                        uni_cell = tf.contrib.rnn.LSTMBlockFusedCell(
                            self.hparams.num_units, use_peephole=False)
                        uni_fused_outputs_op, uni_fused_state_op = uni_cell(
                            uni_inputs, sequence_length=self.sequence_length, dtype=inputs.dtype)
                        encoder_states.append(uni_fused_state_op)
                        outputs.append(uni_fused_outputs_op)
                        if i > 0:
                            uni_fused_outputs_op = uni_fused_outputs_op + uni_inputs
                        uni_inputs = uni_fused_outputs_op

            final_output = None
            # embedding + fw + bw + uni
            n = 3 + self.hparams.num_layers - 1
            scalars = tf.get_variable(
                'scalar', initializer=tf.constant([1 / (n)] * n))
            self.scalars = scalars
            weight = tf.get_variable('weight', initializer=tf.constant(0.001))
            self.weight = weight

            soft_scalars = tf.nn.softmax(scalars)
            for i, output in enumerate(outputs):
                if final_output is None:
                    final_output = soft_scalars[i] * \
                        tf.transpose(output, [1, 0, 2])
                else:
                    final_output = final_output + \
                        soft_scalars[i] * tf.transpose(output, [1, 0, 2])

            self.final_outputs = weight * final_output
            self.final_state = tuple(encoder_states)

    def setup_attention_semantic(self):
        num_units = self.hparams.num_units * \
            2 if self.hparams.double_decoder else self.hparams.num_units
        with tf.variable_scope("attention_semantic") as scope:
            cell = single_rnn_cell(self.hparams.rnn_cell_name, num_units, self.is_training(
            ), self.dropout_keep_prob, self.hparams.weight_keep_drop, self.hparams.variational_dropout)
            attention = tf.contrib.seq2seq.LuongAttention(
                num_units, self.final_outputs, self.sequence_length, scale=True)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention, output_attention=True)
            if 'lstm' in self.hparams.rnn_cell_name.lower():
                h = tf.layers.dense(tf.concat(
                    [state.h for state in self.final_state], axis=-1), num_units, use_bias=True)
                c = tf.layers.dense(tf.concat(
                    [state.c for state in self.final_state], axis=-1), num_units, use_bias=True)
                initial_state = attn_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                    cell_state=tf.contrib.rnn.LSTMStateTuple(c=c, h=h))
            else:
                h = tf.layers.dense(tf.concat(
                    [state for state in self.final_state], axis=-1), num_units, use_bias=True)
                initial_state = attn_cell.zero_state(
                    self.batch_size, dtype=tf.float32).clone(cell_state=h)

            state = initial_state
            memory = self.final_outputs
            memory_length = self.sequence_length
            helper = tf.contrib.seq2seq.TrainingHelper(
                memory,
                memory_length
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell,
                helper=helper,
                initial_state=state
            )
            max_dec_len = tf.reduce_max(
                self.sequence_length, name="max_dec_len")
            outputs, dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=max_dec_len,
                swap_memory=True)
            return outputs.rnn_output

    def setup_clf(self):
        logits = self.setup_attention_semantic()
        if self.hparams.add_transform:
          with tf.variable_scope('cls/predictions'):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
              logits = tf.layers.dense(
                  logits,
                  units=self.hparams.num_units,
                  activation=tf.nn.relu,
                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
              logits = layer_norm(logits)
        with tf.variable_scope("ner", reuse=tf.AUTO_REUSE) as scope:
            # logits = self.final_outputs
            logits = tf.layers.dense(logits,self.hparams.target_label_num)
            with tf.variable_scope("crf"):
                # setup crf layer
                trans = tf.get_variable(
                    "transitions",
                    shape=[self.hparams.target_label_num,
                           self.hparams.target_label_num],
                    initializer=tf.contrib.layers.xavier_initializer())
                if self.hparams.init_transition_path:
                  print("Load init transition weight")
                  init_transition_params = np.load(self.hparams.init_transition_path)
                  mask = np.isinf(init_transition_params)
                  trans = tf.where(mask, x=init_transition_params, y=trans)

                pred_ids, _ = tf.contrib.crf.crf_decode(
                    potentials=logits, transition_params=trans, sequence_length=self.sequence_length)
                if self.hparams.mode != 'inference':
                    self.log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                        inputs=logits,
                        tag_indices=self.target_labels,
                        transition_params=trans,
                        sequence_lengths=self.sequence_length)

            self.final_predict = pred_ids
            if self.hparams.mode in ['train', 'eval']:
                self.accurary = tf.contrib.metrics.accuracy(tf.to_int32(
                    self.final_predict), tf.to_int32(self.target_labels))

    def setup_loss(self):
        self.losses = tf.reduce_mean(-self.log_likelihood)

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(
            self.hparams.checkpoint_dir, tf.get_default_graph())
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy", self.accurary)
        tf.summary.scalar('gN', self.gradient_norm)
        tf.summary.scalar('pN', self.param_norm)
        self.summary_op = tf.summary.merge_all()

    def setup_training(self):
        global_step = self.global_step

        if self.hparams.num_warmup_steps > 0:
          learning_rate = tf.constant(value=self.hparams.learning_rate, shape=[], dtype=tf.float32)

          # Implements linear decay of the learning rate.
          learning_rate = tf.train.polynomial_decay(
              learning_rate,
              global_step,
              self.hparams.num_train_steps,
              end_learning_rate=0.01*self.hparams.learning_rate,
              power=1.0,
              cycle=False)

          # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
          # learning rate will be `global_step/num_warmup_steps * init_lr`.
          if self.hparams.num_warmup_steps > 0:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(self.hparams.num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = self.hparams.learning_rate * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
          self.learning_rate = learning_rate
        else:
          self.learning_rate = tf.Variable(
            self.hparams.learning_rate, dtype=tf.float32, trainable=False)

        params = self.params
        if self.hparams.l2_loss_ratio > 0:
            l2_loss = self.hparams.l2_loss_ratio * \
                tf.add_n([tf.nn.l2_loss(p) for p in params if (
                    'predict_clf' in p.name and 'bias' not in p.name)])
            self.losses += l2_loss

        get_total_param_num(params)

        self.param_norm = tf.global_norm(params)

        gradients = tf.gradients(
            self.losses, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.hparams.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        opt = tf.train.RMSPropOptimizer(self.learning_rate)
        train_op = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)
        if self.hparams.ema:
          with tf.control_dependencies([train_op]):
              train_op = self.ema.apply(params)
        self.train_op = train_op

    def train_clf_one_step(self, sess, source, lengths, targets, add_summary=False, run_info=False):
        feed_dict = {}
        feed_dict[self.source_tokens] = source
        feed_dict[self.sequence_length] = lengths
        feed_dict[self.target_labels] = targets
        if run_info:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, batch_loss, summary, global_step, accuracy, token_num, batch_size = sess.run(
                [self.train_op, self.losses, self.summary_op, self.global_step,
                    self.accurary, self.predict_token_num, self.batch_size],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)

        else:
            _, batch_loss, summary, global_step, accuracy, token_num, batch_size = sess.run(
                [self.train_op, self.losses, self.summary_op, self.global_step,
                    self.accurary, self.predict_token_num, self.batch_size],
                feed_dict=feed_dict
            )
        if run_info:
            self.summary_writer.add_run_metadata(
                run_metadata, 'step%03d' % global_step)
            print("adding run meta for", global_step)

        if add_summary:
            self.summary_writer.add_summary(summary, global_step=global_step)
        return batch_loss, global_step, accuracy, token_num, batch_size

    def eval_clf_one_step(self, sess, source, lengths, targets):
        feed_dict = {}
        feed_dict[self.source_tokens] = source
        feed_dict[self.sequence_length] = lengths
        feed_dict[self.target_labels] = targets

        batch_loss, accuracy, batch_size, predict = sess.run(
            [self.losses, self.accurary, self.batch_size, self.final_predict],
            feed_dict=feed_dict
        )
        return batch_loss, accuracy, batch_size, predict

    def inference_clf_one_batch(self, sess, source, lengths):
        feed_dict = {}
        feed_dict[self.source_tokens] = source
        feed_dict[self.sequence_length] = lengths

        predict = sess.run(
            self.final_predict, feed_dict=feed_dict)
        return predict

class MLMModel(Model):
  def setup_input_placeholders(self):
    self.source_tokens = tf.placeholder(
        tf.int32, shape=[None, None], name='source_tokens')
    self.input_mask = tf.placeholder(tf.int32,shape=[None,None],name='input_mask')
    self.batch_size = tf.shape(self.source_tokens, out_type=tf.int32)[0]
    self.sequence_length = tf.reduce_sum(self.input_mask,axis=-1)
    self.token_num = tf.reduce_sum(self.sequence_length)
    self.masked_lm_positions = tf.placeholder(tf.int32,shape=[None,None],name='masked_lm_positions')
    self.masked_lm_ids = tf.placeholder(tf.int32,shape=[None,None],name='masked_lm_ids')
    self.masked_lm_weights = tf.placeholder(tf.float32, shape=[None, None],name="masked_lm_weights")

    self.global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
    self.embedding_dropout = tf.Variable(
        self.hparams.embedding_dropout, trainable=False)
    self.dropout_keep_prob = tf.Variable(
        self.hparams.dropout_keep_prob, trainable=False)

  def build(self):
    self.setup_input_placeholders()
    self.setup_embedding()
    if self.hparams.encoder == 'gnmt':
        self.gnmt_encoder()
    elif self.hparams.encoder == 'elmo':
        self.elmo_encoder()
    else:
        raise ValueError("Un-supported encoder %s" % self.hparams.encoder)
    self.setup_mlm()

    self.params = tf.trainable_variables()
    if self.hparams.ema:
      self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)

    if self.hparams.mode in ['train', 'eval']:
        self.setup_loss()
    if self.hparams.mode == 'train':
        self.setup_training()
        self.setup_summary()
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

  def setup_mlm(self):
    logits = self.setup_attention_semantic()
    input_tensor = gather_indexes(logits, self.masked_lm_positions)
    with tf.variable_scope('cls/predictions'):
      # We apply one more non-linear transformation before the output layer.
      # This matrix is not used after pre-training.
      with tf.variable_scope("transform"):
        input_tensor = tf.layers.dense(
            input_tensor,
            units=self.hparams.num_units,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        input_tensor = layer_norm(input_tensor)

      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      output_bias = tf.get_variable(
          "output_bias",
          shape=[self.hparams.vocab_size],
          initializer=tf.zeros_initializer())
      logits = tf.matmul(input_tensor, self.embedding, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      label_ids = tf.reshape(self.masked_lm_ids, [-1])
      label_weights = tf.reshape(self.masked_lm_weights, [-1])

      one_hot_labels = tf.one_hot(
          label_ids, depth=self.hparams.vocab_size, dtype=tf.float32)

      # The `positions` tensor might be zero-padded (if the sequence is too
      # short to have the maximum number of predictions). The `label_weights`
      # tensor has a value of 1.0 for every real prediction and 0.0 for the
      # padding predictions.
      per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
      numerator = tf.reduce_sum(label_weights * per_example_loss)
      denominator = tf.reduce_sum(label_weights) + 1e-5
      self.loss = numerator / denominator
      self.masked_lm_example_loss = per_example_loss
      masked_lm_log_probs = tf.reshape(log_probs,[-1, log_probs.shape[-1]])
      masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
      self.accurary = tf.contrib.metrics.accuracy(
            labels=label_ids,
            predictions=masked_lm_predictions,
            weights=label_weights)

  def setup_loss(self):
    self.losses = self.loss

  def train_clf_one_step(self, sess, features, add_summary=False):
    feed_dict = {}
    feed_dict[self.source_tokens] = features['input_ids']
    feed_dict[self.input_mask] = features["input_mask"]
    feed_dict[self.masked_lm_positions] = features["masked_lm_positions"]
    feed_dict[self.masked_lm_ids] = features["masked_lm_ids"]
    feed_dict[self.masked_lm_weights] = features["masked_lm_weights"]
    
    _, batch_loss, summary, global_step, accuracy, token_num = sess.run(
        [self.train_op, self.losses, self.summary_op, self.global_step, self.accurary, self.token_num],
        feed_dict=feed_dict
    )
    if add_summary:
        self.summary_writer.add_summary(summary, global_step=global_step)
    return batch_loss, global_step, accuracy,token_num