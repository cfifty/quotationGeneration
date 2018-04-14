import nltk	
import numpy as np
import os
import csv
import pickle
from nltk.tokenize import TreebankWordTokenizer

class Model():
	def __init__(self,lst=None):
		# load the list of sentences from disk
		if lst == None:
			with open('corpus.pkl','rb') as input:
				self.sentence_lst = pickle.load(input)
		else:
			self.sentence_lst = lst

		self.uni_counts = {}
		self.bi_counts = {'<s>':{}}
		self.tri_counts = {'<s>':{}}

	# add a start token to each sentence and return a list of tokens 
	def add_start(self,lst):
		tokenizer = TreebankWordTokenizer()
		rtn = [["<s>"] + tokenizer.tokenize(entry.lower()) for entry in lst]
		return rtn 

	def gen_end_chars(self):
		end_char = set()
		end_char.add(".")
		end_char.add("!")
		end_char.add("?")
		return end_char

	def train_unigram_model(self):
		for tokens in self.add_start(self.sentence_lst):
			for i in xrange(1,len(tokens)):
				if tokens[i] not in self.uni_counts:
					self.uni_counts[tokens[i]] = 0
				self.uni_counts[tokens[i]] += 1
		return self.uni_counts

	def train_bigram_model(self):
		end_char = self.gen_end_chars()
		for tokens in self.add_start(self.sentence_lst):
			for i in xrange(1,len(tokens)):
				if tokens[i - 1] not in self.bi_counts: self.bi_counts[tokens[i-1]] = {}
				if tokens[i] not in self.bi_counts[tokens[i-1]]: self.bi_counts[tokens[i-1]][tokens[i]] = 0
				self.bi_counts[tokens[i-1]][tokens[i]] += 1
		return self.bi_counts

	def train_trigram_model(self):
		end_char = self.gen_end_chars()
		for tokens in self.add_start(self.sentence_lst):
			for i in xrange(2,len(tokens)):
				if tokens[i-2] not in self.tri_counts: self.tri_counts[tokens[i-2]] = {}
				if tokens[i-1] not in self.tri_counts[tokens[i-2]]: self.tri_counts[tokens[i-2]][tokens[i-1]] = {}
				if tokens[i] not in self.tri_counts[tokens[i-2]][tokens[i-1]]: self.tri_counts[tokens[i-2]][tokens[i-1]][tokens[i]] = 0
				self.tri_counts[tokens[i-2]][tokens[i-1]][tokens[i]] += 1
		return self.tri_counts


	# return the next word from bigram model with probabilities influenced by simple Kneser-Ney smoothing
	def kneser_bigram_prob(self,prev_word):

		# if the prev_word is not seen in the bigrams, it must be some sort of end punctuation
		# this case should never occur because we stop when we see a stop word
		if prev_word not in self.bi_counts: 
			return "."

		all_words_set = set(self.uni_counts.keys())
		bigram_words_set = set(self.bi_counts[prev_word].keys())
		unseen_words = all_words_set - bigram_words_set 

		unseen_word_counts = 0.000027*len(unseen_words)
		bigram_words_counts = {k:(v - 0.5) if v == 1 else (v - 0.75) for k,v in self.bi_counts[prev_word].iteritems()}

		div = unseen_word_counts + sum(bigram_words_counts.values())
		choices = [k for k,v in self.bi_counts[prev_word].iteritems()]
		prob_dist = [v/div for k,v in bigram_words_counts.iteritems()]

		choices.append("UNSEEN")
		prob_dist.append(unseen_word_counts/div)

		chosen_word = np.random.choice(choices,p=prob_dist)
		if chosen_word == "UNSEEN":
			chosen_word = unseen_words.get(np.randdom.randint(0,len(unseen_words)))
		return chosen_word

	# return the next word from trigram model with probabilities influenced by simple Kneser-Ney smoothing
	def kneser_trigram_prob(self,prev_word, mid_word):

		# if our generated bigram is not in the seen trigrams, use a bigram LM instead
		if mid_word not in self.tri_counts[prev_word]:
			return self.kneser_bigram_prob(mid_word)


		all_words_set = set(self.uni_counts.keys())
		print(self.tri_counts[prev_word][mid_word])
		trigram_words_set = set(self.tri_counts[prev_word][mid_word].keys())
		unseen_words = all_words_set - trigram_words_set

		unseen_word_counts = 0.000027*len(unseen_words)
		trigram_words_counts = {k:(v - 0.5) if v == 1 else (v-0.75) for k,v in self.tri_counts[prev_word][mid_word].iteritems()}
		
		div = unseen_word_counts + sum(trigram_words_counts.values())
		choices = [k for k,v in self.tri_counts[prev_word][mid_word].iteritems()]
		prob_dist = [v/div for k,v in trigram_words_counts.iteritems()]

		choices.append("UNSEEN")
		prob_dist.append(unseen_word_counts/div)

		print(prob_dist)

		chosen_word = np.random.choice(choices,p=prob_dist)
		if chosen_word == "UNSEEN":
			chosen_word = unseen_words.get(np.randdom.randint(0,len(unseen_words)))
		return chosen_word


	# use a bigram model to select the next word in a sentence: no smoothing
	def bi_gen_token(self,prev_word):
		next_tokens = self.bi_counts[prev_word]
		prob_dist = []
		divisor = len(self.sentence_lst) if prev_word == "<s>" else self.uni_counts[prev_word]
		for token in next_tokens:
			prob_dist.append(next_tokens[token]/divisor)
		return np.random.choice(next_tokens,p=prob_dist)


	# use a trigram model to select the next word in a sentence: no smoothing
	def tri_gen_token(self,prev_word,mid_word):
		next_tokens = self.tri_counts[prev_word][mid_word]
		prob_dist = []
		divisor = len(self.sentence_lst) if prev_word == "<s>" else self.bi_counts[prev_word]
		for token in next_tokens:
			prob_dist.append(next_tokens[token]/divisor)
		return np.random.choice(next_tokens,p=prob_dist)

	# generate a sentence using bigram probabilities 
	def bigram_sentence(self):
		return_sentence = ""
		end_char_set = self.gen_end_chars()
		prev = "<s>"

		while prev not in end_char_set:
			cur_token = self.kneser_bigram_prob(prev)
			# uncomment to use un-smoothed bigram probabilities
			# cur_token = self.bi_gen_token(prev)
			if cur_token in end_char_set:
				return_sentence += cur_token
				break 
			else: 
				return_sentence += cur_token + " "
			prev = cur_token 
		return return_sentence

	def trigram_sentence(self):
		return_sentence = "" 
		end_char_set = self.gen_end_chars()
		prev = ""
		mid = "<s>"

		while mid not in end_char_set:
			cur_token = self.kneser_bigram_prob(mid) if prev == "" else self.kneser_trigram_prob(prev,mid)
			# uncomment ot use un-smoothed probabilites
			# cur_token = self.bi_gen_token(prev) iv mid == "" else self.kneser_bigram_prob(prev,mid)
			if cur_token in end_char_set:
				return_sentence += cur_token
				break 
			else: 
				return_sentence += cur_token + " " 
			prev = mid 
			mid = cur_token 
		return return_sentence 

''' 
Test Suite to ensure functionality

add_start: 							works!
trin_unigram_model: 				works!
train_bigram_model:					works!
train_trigram_model: 				works!

kneser_bigram_prob: 				works!
kneser_trigram_prob: 				works!

'''
data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data_files'))

# load the data from the trump quotes csv
f = open(data_path+'/trumpquotes.csv','rb')
reader = csv.reader(f)   
trump_quotes = [x[0] for x in reader]

#with open(data_path + '/trumpquotes.pkl','rb') as input:
#	trump_quotes = pickle.load(input)

model = Model(trump_quotes)
model.train_unigram_model()
model.train_bigram_model()
model.train_trigram_model()

#print (model.bigram_sentence())
print (model.trigram_sentence())






