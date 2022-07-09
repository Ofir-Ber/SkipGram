from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from collections import Counter
from nltk import stem
from string import digits
from tqdm import tqdm
import time
from datetime import timedelta, datetime
import json
from sklearn.decomposition import PCA


__authors__ = ['Ofir Berger','Sauvik Chatterjee','Bryan Atok A Kiki', 'Abhinav Singh']
__emails__  = ['b00791562@essec.edu','b00782253@essec.edu','b00792559@essec.edu', 'b00789513@essec.edu']

def text2sentences(path):
	# feel free to make a better tokenization/pre-processing
	sentences = []
	stemmed_word_list = []
	stemmer=stem.snowball.EnglishStemmer()
	with open(path, encoding='utf8') as f:
		for l in f:
			sent = l.lower().split()
			sent_new = [word.translate(str.maketrans('','','''!"#$%&()*+,./:;<=>?@[\]^_`{|}~''')) for word in sent]
			sent_new_1 = [word.translate(str.maketrans('', '', digits)) for word in sent_new] # clean digits
			sent_new_2 = [word.replace('-','') for word in sent_new_1] # clean hyphen 
			sent_new_3 = [word.replace("'s",'') for word in sent_new_2] # clean 's
			sent_new_4 = [word.replace("'t",'') for word in sent_new_3] # clean 't
			sent_new_5 = [word for word in sent_new_4 if len(word)>1] # clean empty 'words'
			sentences.append(sent_new_5)
	# Stemming
	#for s in sentences:
	#	stemmed_sentences = []
	#	for word in s:
	#		stemmed_sentences.append(stemmer.stem(word))
	#	stemmed_word_list.append(stemmed_sentences)
	return sentences

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'])
	return pairs

def drop(u, v):
  return u - v * u.dot(v)/v.dot(u)



class SkipGram:
	def __init__(self, sentences, nEmbed=500, negativeRate=5, winSize = 7, minCount = 3, epochs=10, lr=0.1):
		self.w2id = {} # word to ID mapping
		self.trainset = sentences # set of sentences
		self.minCount = minCount # Minimum occurrence of words
		words = [word for sentence in sentences for word in sentence] # Store every word appearing in the trainset with repetitions
		word_count = dict(Counter(words)) # Return the frequency of words in the set

		# filter words with minCount
		word_count = {word:freq for word, freq in word_count.items() if freq >= self.minCount}
		words = [word for word, freq in word_count.items()]

		self.vocab = sorted(set(words)) # list of valid words
		vocab_size = len(self.vocab) # Size of the vocabulary
		word_count = dict(Counter(words)) # Return the frequency of words in the set
		
		self.freq = np.array([word_count[word] / vocab_size for word in self.vocab]) # frequency of words, to be used for the unigram
		self.nEmbed = nEmbed
		self.negativeRate = negativeRate
		self.winSize = winSize
		self.epochs = epochs
		self.lr = lr # Learning rate
		self.accLoss = 0
		self.trainWords = 0
		self.loss = [] # list that will store the loss values
		self.loss_epoch = [] # list that will store the loss values of a particular epoch

		# Initialization -- > Try others
		#self.centerV = (np.random.rand(vocab_size, nEmbed) - 0.5) / nEmbed # Initialize random vector for center words
		#self.contxtV = (np.random.rand(vocab_size, nEmbed) - 0.5) / nEmbed # Initialize random vector for context words

		# He Initialisation - np.random.randn(a, b) * np.sqrt(2/b)
		#self.centerV = np.random.rand(vocab_size, nEmbed) * np.sqrt(2/nEmbed) # Initialize random vector for center words
		#self.contxtV = np.random.rand(vocab_size, nEmbed) * np.sqrt(2/nEmbed) # Initialize random vector for context words

		self.centerV = np.random.randn(vocab_size, nEmbed)
		self.contxtV = np.random.randn(vocab_size, nEmbed)
        
        
		## Prepare w2id: word to index
		for i, word in enumerate(self.vocab):
			self.w2id[word] = i

		# Create the unigram table
		self.unigram_table = self.compute_unigram_table()

		#print("Initalization completed!")

	def compute_unigram_table(self, exponent=0.75, table_length = int(1e8)):
		""" Compute the unigram table """
		# Length of the unigram table is equal to 1e8
		# This large table will contain every words, repeated as many times as their frequency
		# The negative sampling will be done from this table

		vocab_size = len(self.vocab)
		
		# Normalization factor for the word probabilities = denominator
		norm_factor = sum(
			[np.power(self.freq[i], exponent) for i in range(vocab_size)]
		)

		table = np.array(np.zeros(table_length), dtype=int)
		p = 0 # Cumulative probability
		count = 0

		# iterate over every words
		for i in range(vocab_size):
			p += np.power(self.freq[i], exponent) / norm_factor

			while (count < table_length) and (count/ table_length < p):
				table[count]= i
				count +=1

		np.random.shuffle(table)

		return table


	def sample(self, omit):
        #"""samples negative words, ommitting those in set omit"""
		count = 0
		negWordsId = []
		# Sample negative words until we reach negativeRate
		while count < self.negativeRate:
			negWordId = np.random.choice(self.unigram_table)
			if negWordId not in omit: #selecting only words that are not in omit (i.e. not in the current context)
				negWordsId.append(negWordId)
				count +=1
		return negWordsId
        

	def train(self):
		print("Entering the training phase.")
		for epoch in tqdm(range(self.epochs)):
			start_time = datetime.now()
			self.loss_epoch = []
			print("\n epoch: %d of %d" % (epoch + 1, self.epochs))
			for counter, sent in enumerate(self.trainset):
				sentence = [word for word  in sent if word in self.vocab]

				for wpos, word in enumerate(sentence):
					wIdx = self.w2id[word]
					winsize = np.random.randint(self.winSize) + 1
					start = max(0, wpos - winsize)
					end = min(wpos + winsize + 1, len(sentence))

					for context_word in sentence[start:end]:
						ctxtId = self.w2id[context_word]
						if ctxtId == wIdx: continue
						negativeIds = self.sample({wIdx, ctxtId})
						self.trainWord(wIdx, ctxtId, negativeIds)
						self.trainWords += 1

				if counter % 1000 == 0:
					end_time = datetime.now()
					print(' > training %d of %d' % (counter, len(self.trainset)), '- Duration: {}'.format(end_time - start_time))
				
					self.loss.append(self.accLoss / self.trainWords)
					self.loss_epoch.append(self.accLoss / self.trainWords)
					self.trainWords = 0
					self.accLoss = 0.
			print("Epoch:", epoch + 1, "Loss:", sum(self.loss_epoch))

	def trainWord(self, wordId, contextId, negativeIds):
		vector = self.centerV[wordId]
		ctxtvector = self.contxtV[contextId]
		negvector = self.contxtV[negativeIds]
		z = expit(-np.dot(ctxtvector, vector)) # logistic sigmoid (i.e inverse of the logit function)
		zNeg = - expit(np.dot(negvector, vector))

		## Compute the gradients
		contxtGrad = z * vector
		centerGrad = z * self.contxtV[contextId] + np.dot(zNeg, negvector)
		negGrad = np.outer(zNeg, vector)

		## Gradient descent step
		np.add(vector, centerGrad * self.lr, out=vector);
		np.add(ctxtvector, contxtGrad * self.lr, out=ctxtvector);
		np.add(negvector, negGrad * self.lr, out= negvector);

		## Compute the loss
		z = expit(np.dot(ctxtvector, vector))
		zNeg = expit(-np.dot(negvector, vector))
		self.accLoss -= np.log(z) + np.sum(np.log(zNeg))

		## Update the embeddings
		self.centerV[wordId] = vector
		self.contxtV[contextId] = contxtGrad
		self.contxtV[negativeIds] = negvector


	def save(self,path):
		""" We will save the model in a .zip file"""
		import os
		from zipfile import ZipFile, ZIP_DEFLATED
		from json import dumps

		if "." in path:
			filename_split = path.split('.')
			if filename_split[-1] != "zip":
				filename_split[-1] = "zip"
				path = ".".join(filename_split)
		else:
			path += "/sg.zip"

		zf = ZipFile(path, mode="w", compression=ZIP_DEFLATED)

		## Save the parameters of the model
		model_info = dumps(
			{
				"nEmbed": self.nEmbed,
				"negativeRate": self.negativeRate,
				"winSize": self.winSize,
				"minCount": self.minCount,
				"w2id": self.w2id,
				"epochs": self.epochs,
				"lr": self.lr,
			}, indent = 5
		)

		trainset = dumps(self.trainset, indent=5)
		vocab = dumps(self.vocab, indent=5)

		zf.writestr("model_info.json", model_info)
		zf.writestr("trainset.json", trainset)
		zf.writestr("vocab.json", vocab)

		# Save embeddings data
		np.save("centerV.npy", self.centerV)
		zf.write("centerV.npy")
		os.remove("centerV.npy")
		np.save("contxtV.npy", self.contxtV)
		zf.write("contxtV.npy")
		os.remove("contxtV.npy")
		np.save("freq.npy", self.freq)
		zf.write("freq.npy")
		os.remove("freq.npy")
		np.save("loss.npy", self.loss)
		zf.write("loss.npy")
		os.remove("loss.npy")

		zf.close()



	def similarity(self,word1,word2):
		"""
		computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""

		word1, word2 = word1.lower(), word2.lower()
		if (word1 in self.vocab) and (word2 in self.vocab):
			vec1 = self.centerV[self.w2id[word1]]
			vec2 = self.centerV[self.w2id[word2]]

			# compute the cosine similarity
			cos_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
			score = np.clip(cos_similarity, 0, 1) # limit the value between 0 and 1
		else:
			score = np.random.rand() # assign a random score to the pair if a word is not in the vocabulary
		
		if score <= 1e-4:
			score = 0
		return score


	@staticmethod
	def load(path):
		from json import loads
		from io import BytesIO
		from zipfile import ZipFile

		try:
			zf = ZipFile(path, "r")
		except FileNotFoundError:
			path = path.split('.')
			path[-1] = 'zip'
			path = '.'.join(path)
			zf = ZipFile(path, "r")

		model_info = loads(zf.read("model_info.json"))
		trainset = loads(zf.read('trainset.json'))
		vocab = loads(zf.read('vocab.json'))

		sg = SkipGram(trainset, nEmbed=model_info['nEmbed'], negativeRate=model_info['negativeRate'],
		winSize=model_info['winSize'], minCount=model_info['minCount'], epochs=model_info['epochs'],
		lr=model_info['lr'])

		sg.vocab = vocab
		sg.centerV = np.load(BytesIO(zf.read('centerV.npy')))
		sg.contxtV = np.load(BytesIO(zf.read('contxtV.npy')))
		sg.loss = np.load(BytesIO(zf.read('loss.npy'))).tolist()
		sg.freq = np.load(BytesIO(zf.read('freq.npy')))

		zf.close()

		return sg
    #### Additional Step : Debias #####
    
    # Step 1: Identify the direction of embedding that captures the gender subspace    
	def doPCA(self, pairs, num_components = 10):
        	matrix = []
        	for a, b in pairs:
                	if (a in self.vocab) and (b in self.vocab):
                		center = (self.centerV[self.w2id[a]] + self.centerV[self.w2id[b]])/2
                		matrix.append(self.centerV[self.w2id[a]] - center)
                		matrix.append(self.centerV[self.w2id[a]] - center)
                	else: continue
        	matrix = np.array(matrix)
        	pca = PCA(n_components = num_components)
        	pca.fit(matrix)
        	return pca
    
    # Step 2: Neutralize - to make sure the gender neutral words are zero in gender subspace
    
	def debias(self):
		definitional = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]
		gender_specific_words = ["actress", "actresses", "aunt", "aunts", "bachelor", "ballerina", "barbershop", "baritone", "beard", "beards", "beau", "bloke", "blokes", "boy", "boyfriend", "boyfriends", "boyhood", "boys", "brethren", "bride", "brides", "brother", "brotherhood", "brothers", "bull", "bulls", "businessman", "businessmen", "businesswoman", "chairman", "chairwoman", "chap", "colt", "colts", "congressman", "congresswoman", "convent", "councilman", "councilmen", "councilwoman", "countryman", "countrymen", "czar", "dad", "daddy", "dads", "daughter", "daughters", "deer", "diva", "dowry", "dude", "dudes", "elder_brother", "eldest_son", "estranged_husband", "estranged_wife", "estrogen", "ex_boyfriend", "ex_girlfriend", "father", "fathered", "fatherhood", "fathers", "fella", "fellas", "female", "females", "feminism", "fiance", "fiancee", "fillies", "filly", "fraternal", "fraternities", "fraternity", "gal", "gals", "gelding", "gentleman", "gentlemen", "girl", "girlfriend", "girlfriends", "girls", "goddess", "		godfather", "granddaughter", "granddaughters", "grandfather", "grandma", "grandmother", "grandmothers", "grandpa", "grandson", "grandsons", "guy", "handyman", "he", "heiress", "hen", "hens", "her", "heroine", "hers", "herself", "him", "himself", "his", "horsemen", "hostess", "housewife", "housewives", "hubby", "husband", "husbands", "king", "kings", "lad", "ladies", "lads", "lady", "lesbian", "lesbians", "lion", "lions", "ma", "macho", "maid", "maiden", "maids", "male", "males", "mama", "man", "mare", "maternal", "maternity", "matriarch", "men", "menopause", "mistress", "mom", "mommy", "moms", "monastery", "monk", "monks", "mother", "motherhood", "mothers", "nephew", "nephews", "niece", "nieces", "nun", "nuns", "obstetrics", "ovarian_cancer", "pa", "paternity", "penis", "prince", "princes", "princess", "prostate", "prostate_cancer", "queen", "queens", "salesman", "salesmen", "schoolboy", "schoolgirl", "semen", "she", "sir", "sister", "sisters", "son", "sons", "sorority", "sperm", "spokesman", "spokesmen","spokeswoman", "stallion", "statesman", "stepdaughter", "stepfather", "stepmother", "stepson", "strongman", "stud", "studs", "suitor", "suitors", "teenage_girl", "teenage_girls", "testosterone", "twin_brother", "twin_sister", "uncle", "uncles", "uterus", "vagina", "viagra", "waitress", "widow", "widower", "widows", "wife", "witch", "witches", "wives", "woman", "womb", "women", "younger_brother"]
		gender_direction = self.doPCA(definitional).components_[0]
		specific_set = set(gender_specific_words)
		for i, w in enumerate(self.vocab):
			if w not in specific_set:
            			self.centerV[self.w2id[w]] = drop(self.centerV[self.w2id[w]], gender_direction)

    





if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()
        

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences)
		sg.train()
        
        # Uncomment below line to run the debias before saving the embedding matrix
		# sg.debias()
        
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
        
		df = pd.read_csv(opts.text, sep ='\t', engine='python')
		#df = pd.read_csv(opts.text, sep ='\t', engine='python')
        
		y_pred = []
		y_test = df['similarity']

		for a,b in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
			score = sg.similarity(a,b)
			y_pred.append(score)
			print(score)
            
		#print(np.corrcoef(y_pred,y_test)[0][1])
