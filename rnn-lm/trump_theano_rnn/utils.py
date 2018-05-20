import numpy as np
import csv
import itertools
import nltk
import time
import sys
import operator
import math
import io
import array
from datetime import datetime

sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
unknown_token = "UNKNOWN_TOKEN"

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])
    
def calc_perplexity(model, index_to_word, word_to_index):
    # load the data from the trump test set
    with open('./data/trump_test_set.csv', 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_start_token) for x in sentences]

    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    perplexity_sum = 0.
    N = 0.

    for i in range(len(tokenized_sentences)):
        print("index: " + str(i))
        print("total: " + str(len(tokenized_sentences)))
        partial_sentence = [word_to_index[sentence_start_token]]
        for word in tokenized_sentences[i][1:-1]:
            #print("here is your word " + str(word))
            next_word_probs = model.forward_propagation(partial_sentence)[-1]
            #print("predictions " + str(next_word_probs))
            if word in word_to_index:
                perplexity_sum += -1*math.log(next_word_probs[word_to_index[word]])
                partial_sentence.append(word_to_index[word])
            else:
                perplexity_sum += -1*math.log(next_word_probs[word_to_index[unknown_token]])
                partial_sentence.append(word_to_index[unknown_token])
            N += 1
        print('estimate of perplexity ' + str(math.exp(perplexity_sum/N)))
    return math.exp(perplexity_sum/N)
