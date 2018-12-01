from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift
from sklearn.mixture import GaussianMixture
from gensim.models import Word2Vec
from sklearn.preprocessing import  StandardScaler
import sys
import nltk
import csv
import string
import numpy as np


def readCsvFile(file):
    listOfRawText = []
    with open(file, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter = '\t')
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
        if i % 3 == 2:
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
        sentence = ""
        for word in sentencesList[i][0]:
            sentence += word + " "
        print (sentence)

        wordTokens = nltk.word_tokenize(sentence)
        if removeStopWords:
            wordTokens = [word for word in wordTokens if not word in stopWords]
        if removePunctuation:
            wordTokens = [word for word in wordTokens if not word in string.punctuation]
        resList.append(wordTokens)
    return resList

def getWord2VecEmbedding(sentencesList):
    model = Word2Vec(sentencesList, min_count = 1)
    words = sorted(model.wv.vocab.keys())
    res = StandardScaler().fit_transform([model[w] for w in words])
    print (res)
    return res

def applyCategorizationModel(data):
    clt = DBSCAN(eps = 3, min_samples = 1).fit(data)
    print (clt.labels_)





filePath = sys.argv[1]
listOfRawText = readCsvFile(filePath)
sentencesList = decomposeRawText(listOfRawText)
sentencesList = preprocessing(sentencesList, 'stoplist.txt', True, True)
res = getWord2VecEmbedding(sentencesList)
applyCategorizationModel(res)
