# wod-generator

This code generates CrossFit WODs (workout of the day) with recurrent neural networks (Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) model). The network is trained on 5k WOD samples collected from various crossfit boxes. Despite of small dataset, the results are both fun and satisfying.

[Some collection of the good results can be seen from here](https://dgurkaynak.github.io/wod-generator/).

## Funny results

Of course, not all the results are great. Here are some funny ones:

Zero WOD
```
0 rounds for time 15 minutes
800 mt run
10 sumo deadlift high pull
9 front squat
1 muscle-up
```

Rest WOD
```
5 rounds for time
20 minutes rest
```

Nearly imposibble WOD
```
5 rounds for time 20 minutes
100 double under
50 wall-ball
```

In this WOD, there is no row but the network still wants burpees over the row
```
7 rounds for time
18 dumbbell squat clean
15 burpees over the row
250 mt run
```

Some imaginary exercises
```
brusseane push-up
t-up
touster
lean & jerk
publ-up
dumbell burpee over the bar
hanging ring in
louble under
ode-hand dip
roundstand walk
tempo kim back extension
muscle sprint
pistol squat snatch
over clean
elite push-up
inverted barbell
rest clean
pill-up
```

## Data

All the WODs are collected from:
- crossfit.com
- findawod.com
- crossfitbalabanlevent.blogspot.com

After collecting, all of WODs are syntactically normalized by hand.
That's why the dataset is that small, I wish to feed a lot more WODs but all the sources uses different format,
some uses abbreviation some don't. To get best results, whole dataset should be in the same format.

The dataset can be found in `data/wods.txt`, or better they are in `wod` table in sqlite database located `db.sqlite`.
Feel free to use it.

## Training

Requirements:

- Python 2.7 (not tested in 3)
- Tensorflow > 1.0

```
> cd char-rnn
> python train.py --help

usage: train.py [-h] [--input_file INPUT_FILE] [--rnn_size RNN_SIZE]
                [--num_layers NUM_LAYERS] [--model MODEL]
                [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                [--num_epochs NUM_EPOCHS] [--log_step LOG_STEP]
                [--grad_clip GRAD_CLIP] [--learning_rate LEARNING_RATE]
                [--decay_rate DECAY_RATE]
                [--input_dropout_keep_prob INPUT_DROPOUT_KEEP_PROB]
                [--output_dropout_keep_prob OUTPUT_DROPOUT_KEEP_PROB]
                [--train_root_dir TRAIN_ROOT_DIR] [--vocab_size VOCAB_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Input text file to train on
  --rnn_size RNN_SIZE   The size of RNN hidden state
  --num_layers NUM_LAYERS
                        The number of layers in the RNN
  --model MODEL         RNN model: rnn, gru, lstm, or nas
  --batch_size BATCH_SIZE
                        Batch size
  --seq_length SEQ_LENGTH
                        RNN sequence length
  --num_epochs NUM_EPOCHS
                        Number of epochs for training
  --log_step LOG_STEP   Logging period in terms of iteration
  --grad_clip GRAD_CLIP
                        Clip gradients value
  --learning_rate LEARNING_RATE
                        Learning rate for adam optimizer
  --decay_rate DECAY_RATE
                        Learning rate for adam optimizer
  --input_dropout_keep_prob INPUT_DROPOUT_KEEP_PROB
                        Input dropout keep probability
  --output_dropout_keep_prob OUTPUT_DROPOUT_KEEP_PROB
                        Output dropout keep probability
  --train_root_dir TRAIN_ROOT_DIR
                        Root directory to put the training data
```

## Sampling

```
> cd char-rnn
> python sample.py --help

usage: sample.py [-h] [--data_dir DATA_DIR] [--seperator_char SEPERATOR_CHAR]
                 [--num_sample NUM_SAMPLE] [--save_to_db [SAVE_TO_DB]]
                 [--nosave_to_db]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Training data directory. If empty, latest folder in
                        training/ folder will be used
  --seperator_char SEPERATOR_CHAR
                        WOD item seperator character, default `|`
  --num_sample NUM_SAMPLE
                        The number of WODs to be sampled, default 1
  --save_to_db [SAVE_TO_DB]
                        Should save into sqlite, default false
  --nosave_to_db
```
