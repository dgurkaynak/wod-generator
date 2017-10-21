"""
Inspired from: https://github.com/sherjilozair/char-rnn-tensorflow
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class CharRNNModel():
    def __init__(self, args):
        self.args = args
        if args.model not in ['rnn', 'gru', 'lstm', 'nas']:
            raise Exception("Model `{}` is not supported".format(args.model))

    def inference(self, x, training=True):
        args = self.args

        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)

            if training and (args.output_dropout_keep_prob < 1.0 or args.input_dropout_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell, input_keep_prob=args.input_dropout_keep_prob,
                    output_keep_prob=args.output_dropout_keep_prob)

            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        batch_size = int(x.shape[0]);
        seq_length = int(x.shape[1]);

        # self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        # self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, x)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_dropout_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_dropout_keep_prob)

        inputs = tf.split(inputs, seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        self.final_state = last_state
        self.logit = tf.matmul(output, softmax_w) + softmax_b
        self.prob = tf.nn.softmax(self.logit)

        return self.prob


    def loss(self, batch_x, batch_y):
        self.inference(batch_x, training=True)
        batch_size = int(batch_x.shape[0]);
        seq_length = int(batch_x.shape[1]);

        seq_loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logit],
                [tf.reshape(batch_y, [-1])],
                [tf.ones([batch_size * seq_length])])

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(seq_loss) / batch_size / seq_length

        return self.loss

    def optimize(self, learning_rate = tf.Variable(0.0, trainable=False)):
        var_list = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, var_list), self.args.grad_clip)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)

        return optimizer.apply_gradients(zip(grads, var_list))
