#!/usr/bin/python

import numpy as np


def loadDataSet():
	texts = [
		'my dog has flea problems help please',
		'maybe not take him to dog park stupid',
		'my dalmation is so cute I love him',
		'stop posting stupid worthless garbage',
		'mr licks ate my steak how to stop him',
		'quit buying worthless dog food stupid'
	]
	postingList = [t.split() for t in texts]
	classes = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not
	
	return postingList, classes

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
		
	return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "The word: %s is not in my vocabulary!" % word
	
	return returnVec

def bagOfWords2Vec(vocabList, inputSet):
	# This model takes into account the number of occurrences of each word
	# as opposed to the set version which only flags if a word is present
	# or not in a document, regardless of how many times it appears which
	# conveys some information about the document
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
			
	return returnVec

def trainNB(trainMatrix, trainClasses):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0]) # Each document has the same word list
	
	# Compute the prior probability for abusive P(abusive) -> (number of occurrences/total number of docs)
	# The prior probability for not abusive is just 
	#    P(not_abusive) = 1 - P(abusive) 
	# because it's a two-class problem
	p_abusive = np.sum(trainClasses) / float(numTrainDocs)
	
	# Let's use class/label 
	# P(1) for abusive and P(0) for not abusive

	# Create a vector/list of conditional probabilities for each word
	# Assume the minimum is 1 as opposed to 0 as we want to avoid multiplying
	# by 0 when classifying. The notion here is also that even though we haven't
	# come across a particular word in a class yet, there's always a small chance
	# it can come up, thus probability is never 0.
	p1_num, p0_num = np.ones(numWords), np.ones(numWords)
	p1_denom, p0_denom = 2.0, 2.0
	
	# Tot up the conditionals by going through each document
	for doc, label in zip(trainMatrix, trainClasses):
		if label == 1:
			p1_num += doc
			p1_denom += np.sum(doc)
		else:
			p0_num += doc
			p0_denom += np.sum(doc)
	
	# Finally calculate the conditional values for each word by dividing
	# Note: We want to avoid underflow (when multiplying many small numberes -> 0)
	#       by using log
	p_words_c1 = np.log(p1_num / p1_denom)
	p_words_c0 = np.log(p0_num / p0_denom)
	
	return p_words_c0, p_words_c1, p_abusive

def train(vocab, docs, classes):
	trainMatrix = [bagOfWords2Vec(vocab, doc) for doc in docs]
	return trainNB(trainMatrix, classes)

def classifyNB(x, p0, p1, pClass1):
	# Note: We use sums instead of products as we're using log
	likelihood_1 = np.sum(x * p1)
	likelihood_0 = np.sum(x * p0)
	prior_1 = np.log(pClass1)
	prior_0 = np.log(1.0 - pClass1)
	postProb_1 = likelihood_1 + prior_1
	postProb_0 = likelihood_0 + prior_0
	
	# p_x = SUM [P(x|c_i) * P(c_i)] = SUM [Likelihood for i * Prior for i]
	# Because we're using log we sum it instead
	p_x = np.exp(likelihood_0 + prior_0) + np.exp(likelihood_1 + prior_1)
	# Normalise to get real probabilities (which should sum to 1)
	prob_0_norm = np.exp(postProb_0) / p_x
	prob_1_norm = np.exp(postProb_1) / p_x
	print prob_0_norm, prob_1_norm, '|', postProb_0, postProb_1, '|', p_x
	
	if prob_1_norm > prob_0_norm:
		return 1
	else:
		return 0

def testingNB():
	posts, classes = loadDataSet()
	vocabList = createVocabList(posts)
	p_words_c0, p_words_c1, p1 = train(vocabList, posts, classes)
	
	test1 = ['love', 'my', 'dalmation']
	test1_descriptor = setOfWords2Vec(vocabList, test1)
	
	test2 = ['stupid', 'garbage']
	test2_descriptor = setOfWords2Vec(vocabList, test2)
	
	r1 = classifyNB(test1_descriptor, p_words_c0, p_words_c1, p1)
	r2 = classifyNB(test2_descriptor, p_words_c0, p_words_c1, p1)

	print('{} classified as: {}'.format(test1, r1))
	print('{} classified as: {}'.format(test2, r2))

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def parseSpamAndHam():
	docs, classes, fullText = [], [], []
	
	for i in range(1, 26):
		wordList = textParse(open('data/email/spam/%d.txt' % i).read())
		docs.append(wordList)
		fullText.extend(wordList)
		classes.append(1)

		wordList = textParse(open('data/email/ham/%d.txt' % i).read())
		docs.append(wordList)
		fullText.extend(wordList)
		classes.append(0)
		
	return docs, classes, fullText

def spamTest():
	docs, classes, fullText = parseSpamAndHam()
	
	# Train Naive Bayes Classifier
	vocabList = createVocabList(docs)
	
	# Shuffle documents and create training and test sets
	docInds = range(len(docs))
	np.random.shuffle(docInds)
	trainingInds, testInds = docInds[:40], docInds[40:]
	
	trainDescriptors = np.array([setOfWords2Vec(vocabList, docs[docInd]) for docInd in trainingInds])
	trainClasses = np.array([classes[docInd] for docInd in trainingInds])

	p_words_c0, p_words_c1, p1 = trainNB(trainDescriptors, trainClasses)

	# Test classifier using the test set and calculate error
	errorCount = 0
	
	for docInd in testInds:
		testDescriptor = np.array(setOfWords2Vec(vocabList, docs[docInd]))
		label = classifyNB(testDescriptor, p_words_c0, p_words_c1, p1)
		if label != classes[docInd]:
			errorCount += 1
			print 'Classification error:', docs[docInd]
	
	errorRate = float(errorCount) / float(len(testInds))
	print(' The error rate is: %.2f' % (errorRate))


spamTest()