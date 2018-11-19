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
from thrid_utils import create_embedding

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
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        
        if self.hparams.mode in ['train', 'eval']:
            self.setup_loss()
        if self.hparams.mode == 'train':
            self.setup_training()
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)

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
        shadow_vars = {self.ema.average_name(v):v for v in self.params}
        saver = tf.train.Saver(shadow_vars)
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
                tf.float32, shape=[None, self.hparams.feature_num, self.hparams.target_label_num], name='target_labels')

        self.batch_size = tf.shape(self.source_tokens,out_type=tf.int32)[0]

        self.sequence_length = tf.placeholder(
            tf.int32, shape=[None], name='sequence_length')
        
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.predict_token_num = tf.reduce_sum(self.sequence_length)
        self.embedding_dropout = tf.Variable(self.hparams.embedding_dropout, trainable=False)
        self.dropout_keep_prob = tf.Variable(self.hparams.dropout_keep_prob, trainable=False)

    def setup_embedding(self):
        # load pretrained embedding
        self.embedding = create_embedding(
            "embedding",
            self.hparams.vocab_size,
            self.hparams.embedding_size,
            vocab_file=self.hparams.vocab_file,
            embed_file=self.hparams.embed_file)
        
        if self.hparams.embedding_dropout > 0 and self.is_training():
            vocab_size = tf.shape(self.embedding)[0]
            mask = tf.nn.dropout(tf.ones([vocab_size]),keep_prob=1-self.embedding_dropout) * (1-self.embedding_dropout)
            mask = tf.expand_dims(mask,1)
            self.embedding = mask * self.embedding
        
        self.source_embedding = tf.nn.embedding_lookup(
            self.embedding, self.source_tokens)
        # [20]
        features = tf.range(self.hparams.feature_num,dtype=tf.int32)
        feature_embedding_var = create_embedding("feature_embedding", self.hparams.feature_num, self.hparams.embedding_size)
        # [20 * embedding_size]
        feature_embedding = tf.nn.embedding_lookup(feature_embedding_var, features)
        # [batch * 20 * embedding_size]
        self.feature_embedding = tf.tile(tf.expand_dims(feature_embedding,axis=0),[self.batch_size,1,1])

        if self.is_training():
            self.source_embedding = tf.nn.dropout(
                    self.source_embedding, keep_prob=self.dropout_keep_prob)
            self.feature_embedding = tf.nn.dropout(
                self.feature_embedding, keep_prob=self.dropout_keep_prob)

    def elmo_encoder(self):
        print_out("build elmo encoder")
        with tf.variable_scope("elmo_encoder") as scope:
            inputs = tf.transpose(self.source_embedding,[1,0,2])
            inputs_reverse = _reverse(
                inputs, seq_lengths=self.sequence_length,
                seq_dim=0, batch_dim=1)
            encoder_states = []
            outputs = [tf.concat([inputs,inputs],axis=-1)]
            fw_cell_inputs = inputs
            bw_cell_inputs = inputs_reverse
            for i in range(self.hparams.num_layers):
                with tf.variable_scope("fw_%d" % i) as s:
                    cell = tf.contrib.rnn.LSTMBlockFusedCell(self.hparams.num_units,use_peephole=False)
                    fused_outputs_op, fused_state_op = cell(fw_cell_inputs,sequence_length=self.sequence_length,dtype=inputs.dtype)
                    encoder_states.append(fused_state_op)
                with tf.variable_scope("bw_%d" % i) as s:
                    bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(self.hparams.num_units,use_peephole=False)
                    bw_fused_outputs_op_reverse, bw_fused_state_op = bw_cell(bw_cell_inputs,sequence_length=self.sequence_length,dtype=inputs.dtype)
                    bw_fused_outputs_op = _reverse(
                        bw_fused_outputs_op_reverse, seq_lengths=self.sequence_length,
                        seq_dim=0, batch_dim=1)
                    encoder_states.append(bw_fused_state_op)
                output = tf.concat([fused_outputs_op,bw_fused_outputs_op],axis=-1)
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
            scalars = tf.get_variable('scalar',initializer=tf.constant([1/(n)]*n))
            self.scalars = scalars
            weight = tf.get_variable('weight',initializer=tf.constant(0.001))
            self.weight = weight

            soft_scalars = tf.nn.softmax(scalars)
            for i, output in enumerate(outputs):
                if final_output is None:
                    final_output = soft_scalars[i] * tf.transpose(output,[1,0,2])
                else:
                    final_output = final_output + soft_scalars[i] * tf.transpose(output,[1,0,2])

            self.final_outputs = weight * final_output
            self.final_state = tuple(encoder_states)

    def gnmt_encoder(self):
        print_out("build gnmt encoder")
        with tf.variable_scope("gnmt_encoder") as scope:
            inputs = tf.transpose(self.source_embedding,[1,0,2])
            inputs_reverse = _reverse(
                inputs, seq_lengths=self.sequence_length,
                seq_dim=0, batch_dim=1)
            encoder_states = []
            outputs = [inputs]

            with tf.variable_scope("fw") as s:
                cell = tf.contrib.rnn.LSTMBlockFusedCell(self.hparams.num_units,use_peephole=False)
                fused_outputs_op, fused_state_op = cell(inputs,sequence_length=self.sequence_length,dtype=inputs.dtype)
                encoder_states.append(fused_state_op)
                outputs.append(fused_outputs_op)
            
            with tf.variable_scope('bw') as s:
                bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(self.hparams.num_units,use_peephole=False)
                bw_fused_outputs_op, bw_fused_state_op = bw_cell(inputs_reverse,sequence_length=self.sequence_length,dtype=inputs.dtype)
                bw_fused_outputs_op = _reverse(
                    bw_fused_outputs_op, seq_lengths=self.sequence_length,
                    seq_dim=0, batch_dim=1)
                encoder_states.append(bw_fused_state_op)
                outputs.append(bw_fused_outputs_op)

            with tf.variable_scope("uni") as s:
                uni_inputs = tf.concat([fused_outputs_op,bw_fused_outputs_op],axis=-1)
                for i in range(self.hparams.num_layers-1):
                    with tf.variable_scope("layer_%d" % i) as scope:
                        uni_cell =  tf.contrib.rnn.LSTMBlockFusedCell(self.hparams.num_units,use_peephole=False)
                        uni_fused_outputs_op, uni_fused_state_op = uni_cell(uni_inputs,sequence_length=self.sequence_length,dtype=inputs.dtype)
                        encoder_states.append(uni_fused_state_op)
                        outputs.append(uni_fused_outputs_op)
                        if i > 0:
                            uni_fused_outputs_op = uni_fused_outputs_op + uni_inputs
                        uni_inputs = uni_fused_outputs_op

            final_output = None
            # embedding + fw + bw + uni
            n = 3 + self.hparams.num_layers - 1
            scalars = tf.get_variable('scalar',initializer=tf.constant([1/(n)]*n))
            self.scalars = scalars
            weight = tf.get_variable('weight',initializer=tf.constant(0.001))
            self.weight = weight
            
            soft_scalars = tf.nn.softmax(scalars)
            for i, output in enumerate(outputs):
                if final_output is None:
                    final_output = soft_scalars[i] * tf.transpose(output,[1,0,2])
                else:
                    final_output = final_output + soft_scalars[i] * tf.transpose(output,[1,0,2])

            self.final_outputs = weight * final_output
            self.final_state = tuple(encoder_states)
    
    def setup_attention_semantic(self):
        num_units = self.hparams.num_units * 2 if self.hparams.double_decoder else self.hparams.num_units
        with tf.variable_scope("attention_semantic") as scope:
            cell = single_rnn_cell(self.hparams.rnn_cell_name, num_units, self.is_training(), self.dropout_keep_prob, self.hparams.weight_keep_drop, self.hparams.variational_dropout)
            attention = tf.contrib.seq2seq.LuongAttention(num_units, self.final_outputs, self.sequence_length,scale=True)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention, output_attention=True)
            if 'lstm' in self.hparams.rnn_cell_name.lower():
                h = tf.layers.dense(tf.concat([state.h for state in self.final_state],axis=-1),num_units, use_bias=True)
                c = tf.layers.dense(tf.concat([state.c for state in self.final_state],axis=-1),num_units, use_bias=True)
                initial_state = attn_cell.zero_state(self.batch_size,dtype=tf.float32).clone(cell_state=tf.contrib.rnn.LSTMStateTuple(c=c,h=h))
            else:
                h = tf.layers.dense(tf.concat([state for state in self.final_state],axis=-1),num_units, use_bias=True)
            
                initial_state = attn_cell.zero_state(self.batch_size,dtype=tf.float32).clone(cell_state=h)
            outputs = []
            state = initial_state
            for i in range(self.hparams.feature_num):
                if i > 0: tf.get_variable_scope().reuse_variables()
                inputs = self.feature_embedding[:,i,:]
                cell_output, state = attn_cell(inputs, state)
                if 'lstm' in self.hparams.rnn_cell_name.lower():
                    out_state = tf.concat([state.cell_state.h,cell_output],axis=-1)
                else:
                    out_state = tf.concat([state.cell_state,cell_output],axis=-1)
                outputs.append(out_state)
            return outputs

    def setup_clf(self):
        num_units = self.hparams.num_units * 2 if self.hparams.double_decoder else self.hparams.num_units
        with tf.variable_scope("classification",reuse=tf.AUTO_REUSE) as scope:
            states = self.setup_attention_semantic()
            final_logits = []
            final_predicts = []
            with tf.variable_scope("predict_clf"):
                hidden_layer = tf.layers.Dense(num_units, use_bias=True, activation=tf.nn.relu)
                output_layer = tf.layers.Dense(self.hparams.target_label_num)

                for i,state in enumerate(states):
                    semantic = hidden_layer(state)
                    logits = output_layer(semantic)

                    final_logits.append(logits)
                    predict = tf.argmax(logits,axis=-1)
                    predict = tf.one_hot(predict,self.hparams.target_label_num)
                    final_predicts.append(predict)

            self.final_logits = tf.concat([tf.expand_dims(l,1) for l in final_logits],axis=1)
            self.final_predict = tf.concat([tf.expand_dims(p,1) for p in final_predicts],axis=1)
            if self.hparams.mode in ['train','eval']:
                self.accurary = tf.contrib.metrics.accuracy(tf.to_int32(self.final_predict),tf.to_int32(self.target_labels))

    def setup_loss(self):
        if self.hparams.focal_loss > 0:
            self.gamma = tf.Variable(self.hparams.focal_loss,dtype=tf.float32, trainable=False)
            label_losses = focal_loss(self.target_labels, self.final_logits, self.gamma)
        else:
            label_losses = tf.losses.softmax_cross_entropy(onehot_labels=self.target_labels, logits=self.final_logits, reduction=tf.losses.Reduction.MEAN)
        self.losses = label_losses

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
        # learning rate decay
        if self.hparams.decay_schema == 'exp':
            self.learning_rate = tf.train.exponential_decay(self.hparams.learning_rate, self.global_step,
                                                            self.hparams.decay_steps, 0.96, staircase=True)
        else:
            self.learning_rate = tf.Variable(
                self.hparams.learning_rate, dtype=tf.float32, trainable=False)

        params = self.params
        if self.hparams.l2_loss_ratio > 0:
            l2_loss = self.hparams.l2_loss_ratio * tf.add_n([tf.nn.l2_loss(p) for p in params if ('predict_clf' in p.name and 'bias' not in p.name)])
            self.losses += l2_loss

        get_total_param_num(params)

        self.param_norm = tf.global_norm(params)

        gradients = tf.gradients(self.losses, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.hparams.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        opt = tf.train.RMSPropOptimizer(self.learning_rate)
        train_op = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)
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
            [self.train_op, self.losses, self.summary_op, self.global_step, self.accurary, self.predict_token_num, self.batch_size],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)
            
        else:
            _, batch_loss, summary, global_step, accuracy, token_num, batch_size = sess.run(
                [self.train_op, self.losses, self.summary_op, self.global_step, self.accurary, self.predict_token_num, self.batch_size],
                feed_dict = feed_dict
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

        batch_loss, accuracy,batch_size, predict = sess.run(
            [self.losses, self.accurary,self.batch_size, self.final_predict],
            feed_dict = feed_dict
        )
        return batch_loss, accuracy,batch_size,predict

    def inference_clf_one_batch(self, sess, source, lengths):
        feed_dict = {}
        feed_dict[self.source_tokens] = source
        feed_dict[self.sequence_length] = lengths

        predict,logits = sess.run([self.final_predict, tf.nn.softmax(self.final_logits)], feed_dict=feed_dict)
        return predict, logits
