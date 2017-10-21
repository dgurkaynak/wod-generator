import collections
import numpy as np


class BatchTextLoader():
    def __init__(self, input_file):
        self.input_file = input_file
        self.pointer = 0

        self.process()

    def process(self):
        with open(self.input_file, 'r') as file:
            data = file.read()
            counter = collections.Counter(data)
            count_pairs = sorted(counter.items(), key=lambda x: -x[1])
            self.chars, _ = zip(*count_pairs)
            self.vocab_size = len(self.chars)
            self.vocab = dict(zip(self.chars, range(len(self.chars))))
            self.tensor = np.array(list(map(self.vocab.get, data)))

    def next_batch(self, batch_size, seq_length):
        length = batch_size * seq_length
        x_data = self.tensor[self.pointer:(self.pointer + length)].reshape(batch_size, -1)
        y_data = self.tensor[(self.pointer + 1):(self.pointer + length + 1)].reshape(batch_size, -1)
        self.pointer += length
        return x_data, y_data

    def reset_pointer(self):
        self.pointer = 0
