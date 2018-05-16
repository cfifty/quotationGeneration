import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *
import RNN_numpy as RNN
from timeit import default_timer as timer

def preprocess(path):
	global index_to_word
	global vocab_size 
	global word_to_index

	sentence_start_token = "<s>"
	sentence_end_token = "</s>"

	# read in the data and add start and end tokens to the quotations
	with open('data/'+path,'rb') as f:
		reader = csv.reader(f,skipinitialspace=True)
		sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
		sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
	print "Parsed %d sentences." % (len(sentences))

	# tokenize the sentences into words 
	tokenized_sentences = [["<s>"] + nltk.word_tokenize(sent[3:-4]) + ["</s>"] for sent in sentences]
	
	# determine word frequencies
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	vocab_size = len(word_freq.items())
	print "Found %d unique words tokens." % len(word_freq.items())

	# Get the most common words and build an index -> word and word -> index lookup list
	vocab = word_freq.most_common()
	index_to_word = [x[0] for x in vocab]
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

	print "\nExample sentence: '%s'" % sentences[0]
	print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
	
	# Create the training data
	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

	return X_train,y_train


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1



def test_forward():
	X_train, y_train = preprocess('ciceroquotes.csv')
	model = RNN.RNNNumpy(vocab_size)
	o, s = model.forward_propagation(X_train[10])
	
	print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in X_train[10]]), X_train[10])
	print "here is your output shape " + str(o.shape)
	print "vocab size " + str(vocab_size)

def test_predict():
	X_train, y_train = preprocess('ciceroquotes.csv')
	model = RNN.RNNNumpy(vocab_size)
	predictions = model.predict(X_train[10])

	print "here is your predictions shape " + str(predictions.shape)
	print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in predictions]), predictions)


def test_loss_random_pred():
	X_train, y_train = preprocess('ciceroquotes.csv')
	model = RNN.RNNNumpy(vocab_size)

	# Limit to 1000 examples to save time
	print "Expected Loss for random predictions: %f" % np.log(vocab_size)
	print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])

def check_gradient():
	# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
	grad_check_vocab_size = 100
	np.random.seed(10)
	model = RNN.RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
	model.gradient_check([0,1,2,3], [1,2,3,4])
	print "done"

def SGD_step():
	X_train, y_train = preprocess('ciceroquotes.csv')
	np.random.seed(10)
	model = RNN.RNNNumpy(vocab_size)

	start = timer()
	model.sgd_step(X_train[10], y_train[10], 0.005)
	end = timer()
	print("fractional seconds for a step " + str((end - start)))

def generate_sentence(model):
	# a sentence begins wiht a start token
	new_sen = [word_to_index['<s>']]

	# repeat until we get an end token
	while not new_sen[-1] == word_to_index['</s>']:
		if len(new_sen) > 50:
			break

		# get the calculated output from the forward_propagation function
		next_word_probs = model.forward_propagation(new_sen)[0]

		# run a multinomial distribution over the last output layer to select a next word weighted by its probability
		samples = np.random.multinomial(1, next_word_probs[-1])

		# get the index of our 1-hot vector => the index into word lookup table
		sampled_word = np.argmax(samples)

		# append the predicted word and loop up with the building sentence
		new_sen.append(sampled_word)

	# strip the start and end characters from the final sentence
	sentence_str = [index_to_word[x] for x in new_sen[1:-1]]
	return sentence_str




X_train, y_train = preprocess('ciceroquotes.csv')
'''
model = RNN.RNNNumpy(vocab_size)
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=1000, evaluate_loss_after=15)

gen_sentence_lst = []
for i in range(100):
	tmp = " ".join(generate_sentence(model))
	gen_sentence_lst.append(str(tmp))

print(gen_sentence_lst)
with open('rnn_cicero_gen.csv', "wb") as f:
	for index,quotation in enumerate(gen_sentence_lst):
		if index == len(gen_sentence_lst) -1:
			f.write('"' + quotation + '"')
		else:
			f.write('"' + quotation + '",\n')
'''

