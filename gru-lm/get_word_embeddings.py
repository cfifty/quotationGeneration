import gensim, logging
import csv
import nltk
import operator
import numpy as np
import pickle

import itertools


SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
vocabulary_size=2000


word_to_index = []
index_to_word = []

# Read the data and append SENTENCE_START and SENTENCE_END tokens
with open('data/ciceroquotes.csv', 'rt') as f:
    reader = csv.reader(f)
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
    sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))
   

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size]
print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
index_to_word = [x[0] for x in sorted_vocab]
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

# train word2vec on the two sentences
model = gensim.models.Word2Vec(tokenized_sentences,size=128, min_count=1)

# summarize vocabulary
words = list(model.wv.vocab)


# create a numpy array in the order of the vocab
vocab_embeddings = np.ones((len(vocab),128))
for row in range(len(vocab_embeddings)):
	print("word " + str(index_to_word[row]) + " in word_embeddings")
	vocab_embeddings[row] = model[str(index_to_word[row])]

vocab_embeddings = vocab_embeddings.T
print(vocab_embeddings)
with open('word_embeddings.pkl','wb') as f:
	pickle.dump(vocab_embeddings,f,protocol=0) # protocol 0 is printable ASCII

print("here is the size of your vocab " + str(len(index_to_word)))
print("here is your serialized numpy vec shape " + str(vocab_embeddings.shape))

