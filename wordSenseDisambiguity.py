from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift
from sklearn.mixture import GaussianMixture
import gensim
import os
from sklearn.preprocessing import  StandardScaler
import sys
import nltk
import string
import numpy as np
import re
import glob

def readTsvDirectroy(path):
    corpusDict = {}
    for filename in glob.glob(os.path.join(path, '*.tsv')):
        with open(filename, 'r') as tsvFile:
            keyWord = re.split(r'/|\.' , filename)[1]
            corpusDict[keyWord] = []
            fileContent = tsvFile.readlines()
            for i in range(1, len(fileContent)):
                fileContent[i] = fileContent[i].rstrip('\n')
                splittedPart = fileContent[i].split('\t')
                corpusDict[keyWord].append(splittedPart)
    return corpusDict

def decomposeRawText(listOfRawText):
    sentencesList = []
    eachSentence = []
    for i in range(300):

        eachSentence.append(listOfRawText[i])
        if i % 3 == 2:
            sentencesList.append(eachSentence)
            eachSentence = []
    return sentencesList

def preprocessing(corpusDict, key, stopWordsFileName, removeStopWords, removePunctuation):
    sentenceList = [rawText[4] for rawText in corpusDict[key]]
    resList = list()
    stopWords = []
    if removeStopWords:
        infile = open(stopWordsFileName, "r")
        stopWordsFile = infile.read().replace("\n", " ")
        stopWords = nltk.word_tokenize(stopWordsFile)

    for i in range(len(sentenceList)):
        wordTokens = nltk.word_tokenize(sentenceList[i])
        if removeStopWords:
            wordTokens = [word for word in wordTokens if not word.lower() in stopWords]
        if removePunctuation:
            wordTokens = [word for word in wordTokens if not word in string.punctuation]
        resList.append(wordTokens)

    for i in range(len(corpusDict[key])):
        corpusDict[key][i][4] = resList[i]
    return key, corpusDict[key]


def getWord2VecEmbedding(key, processedList):
    list1 = [rawText[4] for rawText in processedList]
    inputList = []
    for i in range(len(list1)):
        index1 =list1[i].index(processedList[i][3])
        eachInput = []
        for j in range(index1 - 2, index1 + 3):
            if j >= 0 and j < len(list1[i]):
                eachInput.append(list1[i][j])
        inputList.append(eachInput)
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    model.save("word2vec.model")
    resList = []
    for sentence in inputList:
        res = []
        for w in sentence:
            try:
                res.append(model[w])
            except KeyError:
                continue
        resList.append(sum(res)/len(res))
    resList = StandardScaler().fit_transform(resList)
    return key, resList


def applyCategorizationModel(data):
    clt = DBSCAN(eps = 22.5, min_samples = 1).fit(data)
    senses = {}
    id = 1
    for ele in clt.labels_:
        key = "sense" + str(int(ele)+1)
        if key not in senses:
            senses[key] = [id]
        else:
            senses[key].append(id)
        id += 1
    return senses


directoryPath = sys.argv[1]
corpusDict = readTsvDirectroy(directoryPath)
key, processedList = preprocessing(corpusDict,'force', 'stoplist.txt', True, True)
key, res = getWord2VecEmbedding(key, processedList)
print(applyCategorizationModel(res))
