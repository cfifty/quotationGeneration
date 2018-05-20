#! /usr/bin/env python

import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from gru_theano import GRUTheano

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "3000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "200"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/trumpquotes.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "3000"))

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)



# compute the perplexity of a pre-loaded model
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)
model = load_model_parameters_theano('./data/gru-theano-2018-05-20-15-21-21.npz')
#print("here is your perplexity " + str(calc_perplexity(model,index_to_word,word_to_index)))
generate_sentences(model, 100, index_to_word, word_to_index)




# uncomment to train a new model
'''
# Load data
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

# Build model
model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

# Print SGD step time
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  global loss_lst 
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(x_train[:10000], y_train[:10000])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss: %f" % loss)
  time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  save_model_parameters_theano(model,"./data/gru-theano-%s.npz" % (time))
  print("\n")

  loss_lst.append((num_examples_seen,loss))

  with open("gru_losses.csv","wb") as f:
    for num_examples,loss in loss_lst:
        f.write("num_examples: " + str(num_examples) + " : loss: " + str(loss) + ",\n")

  sys.stdout.flush()

loss_lst = []
for epoch in range(NEPOCH):
  train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback)

'''