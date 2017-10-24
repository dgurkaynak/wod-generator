import os, sys
import numpy as np
import tensorflow as tf
import cPickle as pickle
from model import CharRNNModel
from loader import BatchTextLoader
import sqlite3


DATA_DIR = '../training/char_rnn_20171022_112745'
SEPERATOR_CHAR = '|'
NUM_WOD_SAMPLE = 1
SAVE_TO_DB = False


# Helper class to access dict members with dot notation
class dot_dict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Connect to db
connection = sqlite3.connect('../db.sqlite')
connection.text_factory = str
c = connection.cursor()

# Read options
flags_file_path = os.path.join(DATA_DIR, 'flags.pkl')
flags_file = open(flags_file_path, 'r')
flags = dot_dict(pickle.load(flags_file))
flags_file.close()

# Read loader data
loader_data_file_path = os.path.join(DATA_DIR, 'loader_data.pkl')
loader_data_file = open(loader_data_file_path, 'r')
loader_data = pickle.load(loader_data_file)
loader_data_file.close()

# Create model
x = tf.placeholder(tf.int32, [1, 1])
model = CharRNNModel(flags)
model.inference(x, training=False)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(os.path.join(DATA_DIR, 'checkpoint'))

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

        prime = SEPERATOR_CHAR
        sampling_type=1
        state = sess.run(model.cell.zero_state(1, tf.float32))

        acc = ''
        char = prime[-1]
        wod_i = 0

        def reset_model_state():
            char = prime[-1]

            for char in prime[:-1]:
                input = np.zeros((1, 1))
                input[0, 0] = loader_data.vocab[char]
                feed = {x: input, model.initial_state: state}
                [state] = sess.run([model.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        reset_model_state()

        while (True):
            input = np.zeros((1, 1))
            input[0, 0] = loader_data.vocab[char]
            feed = {x: input, model.initial_state: state}
            [prob, state] = sess.run([model.prob, model.final_state], feed)
            p = prob[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = loader_data.chars[sample]

            if pred == SEPERATOR_CHAR:
                reset_model_state()

                acc = acc.strip()
                if acc == '': continue

                wod_i += 1

                print("\n{}\n".format(acc))
                if SAVE_TO_DB: c.execute('INSERT INTO sample (content, processed) VALUES (?, 0)', (acc,))
                acc = ''

                if wod_i == NUM_WOD_SAMPLE: break
            else:
                acc += pred
                char = pred

        connection.commit()
        connection.close()
