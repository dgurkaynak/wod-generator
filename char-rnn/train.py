import os, sys
import numpy as np
import tensorflow as tf
import datetime
import cPickle as pickle
from model import CharRNNModel
from loader import BatchTextLoader


tf.app.flags.DEFINE_string('input_file', '../data/wods.txt', 'Input text file to train on')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'The size of RNN hidden state')
tf.app.flags.DEFINE_integer('num_layers', 2, 'The number of layers in the RNN')
tf.app.flags.DEFINE_string('model', 'lstm', 'RNN model: rnn, gru, lstm, or nas')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_integer('seq_length', 50, 'RNN sequence length')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')
tf.app.flags.DEFINE_float('grad_clip', 5.0, 'Clip gradients value')
tf.app.flags.DEFINE_float('learning_rate', 0.002, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('decay_rate', 0.97, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('input_dropout_keep_prob', 1.0, 'Input dropout keep probability')
tf.app.flags.DEFINE_float('output_dropout_keep_prob', 1.0, 'Output dropout keep probability')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('vocab_size', 0, 'Do not set this option, it will be inferred automatically')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Batch text loader
    loader = BatchTextLoader(FLAGS.input_file)
    FLAGS.__flags['vocab_size'] = loader.vocab_size

    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('char_rnn_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)

    # Write options
    flags_file_path = os.path.join(train_dir, 'flags.pkl')
    flags_file = open(flags_file_path, 'w')
    pickle.dump(FLAGS.__flags, flags_file)
    flags_file.close()

    # Write loader stuff
    loader_data_file_path = os.path.join(train_dir, 'loader_data.pkl')
    loader_data_file = open(loader_data_file_path, 'w')
    pickle.dump(loader, loader_data_file)
    loader_data_file.close()

    # Placeholders
    x = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.seq_length])
    y = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.seq_length])
    learning_rate = tf.Variable(0.0, trainable=False)

    # Model
    model = CharRNNModel(FLAGS)
    loss = model.loss(x, y)
    train_op = model.optimize(learning_rate)

    # Summaries
    tf.summary.histogram('logit', model.logit)
    tf.summary.scalar('loss', model.loss)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    num_batches_per_epoch = np.floor(len(loader.tensor) / (FLAGS.batch_size * FLAGS.seq_length)).astype(np.int16)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        # model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1

            sess.run(tf.assign(learning_rate, FLAGS.learning_rate * (FLAGS.decay_rate ** epoch)))
            # loader.reset_pointer()
            state = sess.run(model.initial_state)

            # Start training
            while step < num_batches_per_epoch:
                batch_xs, batch_ys = loader.next_batch(FLAGS.batch_size, FLAGS.seq_length)

                # Hmm
                feed = {x: batch_xs, y: batch_ys}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                sess.run([model.final_state, train_op], feed)

                # Logging
                if step % FLAGS.log_step == 0:
                    s = sess.run(merged_summary, feed)
                    train_writer.add_summary(s, epoch * num_batches_per_epoch + step)

                step += 1

            # Epoch completed
            # Reset the dataset pointers
            loader.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            #save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_path)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

if __name__ == '__main__':
    tf.app.run()
