from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift
from sklearn.mixture import GaussianMixture
from gensim.models import Word2Vec
import nltk
import csv
import string
import numpy as np


def readCsvFile(file):
    listOfRawText = []
    with open(file, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        for row in csvReader:
            listOfRawText.append(row)
    csvFile.close()
    #for row in listOfSentence:
    #    print (row)
    return listOfRawText

def decomposeRawText(listOfRawText):
    #decompose text
    sentencesList = []
    eachSentence = []
    for i in range(len(listOfRawText)):
        eachSentence.append(listOfRawText[i])
        if i == 2:
            sentencesList.append(eachSentence)
            eachSentence = []
    return sentencesList

def preprocessing(sentencesList, stopWordsFileName, removeStopWords, removePunctuation):
    resList = list()
    stopWords = []
    if removeStopWords:
        infile = open(stopWordsFileName, "r")
        stopWordsFile = infile.read().replace("\n", " ")
        stopWords = nltk.word_tokenize(stopWordsFile)

    for i in range(len(sentencesList)):
        wordTokens = nltk.word_tokenize(sentencesList[i])
        if removeStopWords:
            wordTokens = [word for word in wordTokens if not word in stopWords]
        if removePunctuation:
            wordTokens = [word for word in wordTokens if not word in string.punctuation]
        resList.append(wordTokens)
    return resList

def getWord2VecEmbedding(sentencesList):
    res = Word2Vec(sentencesList.sents())
    return res





listOfRawText = readCsvFile("fileName")
sentencesList = decomposeRawText(listOfRawText)
sentencesList = preprocessing(sentencesList, True, True, 'stoplist.txt')
res = getWord2VecEmbedding(sentencesList)
