from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift
import gensim
import os
import sys
import nltk
import string
import numpy as np
import re
import glob
from sklearn.preprocessing import normalize
import subprocess

#model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

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
        eachInput = []
        for j in range(len(list1[i])):
            eachInput.append(list1[i][j])
        inputList.append(eachInput)
    resList = []
    for sentence in inputList:
        res = []
        for w in sentence:
            try:
                res.append(model[w])
            except KeyError:
                continue
        resList.append(sum(res)/len(res))
    resList = normalize(resList, norm='l2', axis=0, copy=True, return_norm=False)
    totaldis = 0
    count = 0
    for i in range(0, len(resList)-1):
        for j in range (i+1, len(resList)):
            a = np.array(resList[i])
            b = np.array(resList[j])
            totaldis += np.linalg.norm(a-b)
            count += 1
    avgdis = totaldis/count
    #print("average distance: ", avgdis)
    #print(resList)
    #resList = StandardScaler().fit_transform(resList)
    return key, resList, avgdis


def applyCategorizationModel(data, avgdis):
    clt = DBSCAN(eps = avgdis, min_samples = 1).fit(data)
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

def applyCategorizationModelNew(data, avgdis, minSample, gammaValues):
    clt = DBSCAN(eps = avgdis * gammaValues, min_samples = minSample).fit(data)
    #clt = AffinityPropagation(damping = minSample).fit(data)
    #clt = MeanShift(bin_seeding=True).fit(data)
    #clt = MeanShift().fit(data)
    senseList = [num + 1 for num in clt.labels_]
    return senseList

def applyCategorizationModelMeanShift(data):
    clt = MeanShift().fit(data)
    #clt = AffinityPropagation(damping = minSample).fit(data)
    senseList = [num + 1 for num in clt.labels_]
    return senseList

def applyCategorizationModelAffinityPropagation(data):
    clt = AffinityPropagation().fit(data)
    senseList = [num + 1 for num in clt.labels_]
    return senseList

def buildWord2VecEmbedding(corpusDict, key):
    k, processedList = preprocessing(corpusDict, key, 'stoplist.txt', True, True)
    k, res, avgdis = getWord2VecEmbedding(key, processedList)
    return res, avgdis

def buildResList(corpusDict, minSample, gammaValues):
    print ("Start excecuted for minSample = " + str(minSample) + "; gammaValues = " + str(gammaValues))
    resLists = []
    for key, lists in corpusDict.items():
        #print ("start to build the file " + key + ".tsv")

        res, avgdis = buildWord2VecEmbedding(corpusDict, key)
        senseList = applyCategorizationModelNew(res, avgdis, minSample, gammaValues)
        for i in range(len(lists)):
            pos = lists[i][1] + '.' + lists[i][2]
            id = lists[i][0]
            sense = senseList[i]
            resLists.append([pos, id, sense])

    return resLists

def buildResListMeanShift(corpusDict):
    resLists = []
    for key, lists in corpusDict.items():
        # print ("start to build the file " + key + ".tsv")
        res, avgdis = buildWord2VecEmbedding(corpusDict, key)
        senseList = applyCategorizationModelMeanShift(res)
        for i in range(len(lists)):
            pos = lists[i][1] + '.' + lists[i][2]
            id = lists[i][0]
            sense = senseList[i]
            resLists.append([pos, id, sense])
    return resLists

def buildResListAffinityPropagation(corpusDict):
    resLists = []
    for key, lists in corpusDict.items():
        # print ("start to build the file " + key + ".tsv")
        res, avgdis = buildWord2VecEmbedding(corpusDict, key)
        senseList = applyCategorizationModelAffinityPropagation(res)
        for i in range(len(lists)):
            pos = lists[i][1] + '.' + lists[i][2]
            id = lists[i][0]
            sense = senseList[i]
            resLists.append([pos, id, sense])
    return resLists


def outputSenses(resLists, minSample):
    with open('senses.out', 'w') as out:
        for i in range(len(resLists)):
            out.write(str(resLists[i][0]) + ' ' + str(resLists[i][1]) + ' ' + str(resLists[i][2]) + '/1.0\n')

def outputSensesFinal(resLists):
    with open('finalResult.out', 'w') as out:
        for i in range(len(resLists)):
            out.write(str(resLists[i][0]) + ' ' + str(resLists[i][1]) + ' ' + str(resLists[i][2]) + '/1.0\n')


def calculateHarmonicMean(bCubed, nmi):
  if bCubed == 0 or nmi == 0:
    return 0
  else:
    return 2/(1/bCubed + 1/nmi)



#modify the code from bert_embedding.py
def find_performance_string(output_string):
  lines = output_string.splitlines()
  for line in reversed(lines):
    if line[:3] == 'all':
      tab_char = line.rfind('\t') # find last tab character
      # get number: start from one past tab, go to all but last \n char
      score = float(line[(tab_char):].strip())
      return score
  print('could not find add')
  return None

#modify the code from bert_embedding.py
def hyperparameter_search_dbscan(trial_directory_path, eps_vals, min_samples_vals, gamma_vals):
    trial_data_dir = 'semeval-2012-task-13-trial-data'
    best_b_cubed = 0
    best_nmi = 0
    best_harmonic_mean = 0
    best_eps = eps_vals[0]
    best_min_samples = min_samples_vals[0]
    best_gamma = gamma_vals[0]

    with open('hyperparameter_results.txt', 'w') as out:
        for eps in eps_vals:
            for min_samples in min_samples_vals:
                for gamma in gamma_vals:
                    #print(
                    #    'eps: ' + str(eps) + '\t' + 'min_samples: ' + str(min_samples) + '\t' + 'gamma: ' + str(gamma))

                    if os.path.isfile('senses.out'):
                        os.remove('senses.out')

                    corpusDict = readTsvDirectroy(trial_directory_path)
                    # print ("newMinSample " + str(minSample) + ":")
                    resLists = buildResList(corpusDict, min_samples, gamma)
                    outputSenses(resLists, min_samples)

                    out.write('############################################################\n')
                    out.write(('epsilon = ' + str(eps) + '\n'))
                    out.write(('min_samples = ' + str(min_samples) + '\n'))
                    out.write('gamma = ' + str(gamma) + '\n')
                    out.write('############################################################\n')

                    result = subprocess.check_output(
                        ['java', '-jar', trial_data_dir + '/evaluation/unsupervised/fuzzy-b-cubed.jar',
                         trial_data_dir + '/evaluation/keys/gold-standard/trial.gold-standard.key', 'senses.out'])
                    b_cubed = find_performance_string(result.decode('utf-8'))
                    out.write('b-cubed: ' + str(b_cubed))
                    out.write('\n')

                    result = subprocess.check_output(
                        ['java', '-jar', trial_data_dir + '/evaluation/unsupervised/fuzzy-nmi.jar',
                         trial_data_dir + '/evaluation/keys/gold-standard/trial.gold-standard.key', 'senses.out'])
                    nmi = find_performance_string(result.decode('utf-8'))
                    out.write('nmi: ' + str(nmi))
                    out.write('\n')

                    hm = calculateHarmonicMean(b_cubed, nmi)
                    out.write('harmonic mean: ' + str(hm))
                    out.write('\n')

                    if (hm > best_harmonic_mean):
                        best_b_cubed = b_cubed
                        best_nmi = nmi
                        best_harmonic_mean = hm
                        best_eps = eps
                        best_min_samples = min_samples
                        best_gamma = gamma

        out.write('best performance:\n')
        out.write('eps: ' + str(best_eps) + '\n')
        out.write('min_samples: ' + str(best_min_samples) + '\n')
        out.write('gamma: ' + str(gamma) + '\n')
        out.write('b_cubed: ' + str(best_b_cubed) + '\n')
        out.write('nmi: ' + str(best_nmi) + '\n')
        out.write('harmonic mean: ' + str(best_harmonic_mean) + '\n')

    print('best performance:')
    print('eps: ' + str(best_eps))
    print('min_samples: ' + str(best_min_samples))
    print('gamma: ' + str(best_gamma))
    print('b_cubed: ' + str(best_b_cubed))
    print('nmi: ' + str(best_nmi))
    print('harmonic mean: ' + str(best_harmonic_mean))

    return best_eps, best_min_samples, best_gamma


trialDirectoryPath = sys.argv[1]
testDirectoryPath = sys.argv[2]
method = sys.argv[3]

if  method == 'DBSCAN':
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    min_vals = np.arange(1, 10, 1)
    gamma_vals = np.arange(0.2, 1.2, 0.1)
    best_eps, best_min_samples, best_gamma = hyperparameter_search_dbscan(trialDirectoryPath, [None],min_vals,gamma_vals)

    # run test data
    print("Start run test")
    corpusDict = readTsvDirectroy(testDirectoryPath)
    resLists = buildResList(corpusDict, best_min_samples, best_gamma)
    outputSensesFinal(resLists)

elif method == 'MeanShift':
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    # run test data
    print("Start run test")
    corpusDict = readTsvDirectroy(testDirectoryPath)
    resLists = buildResListMeanShift(corpusDict)
    outputSensesFinal(resLists)

elif method == 'AffinityPropagation':
    # run test data
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    print("Start run test")
    corpusDict = readTsvDirectroy(testDirectoryPath)
    resLists = buildResListAffinityPropagation(corpusDict)
    outputSensesFinal(resLists)

else:
    print ("Error! The third argument must be one of the following three:")
    print ('DBSCAN, MeanShift, AffinityPropagation')
    sys.exit(-1)

test_data_dir = 'SemEval-2013-Task-13-test-data'
result1 = subprocess.check_output(
                        ['java', '-jar', test_data_dir + '/scoring/fuzzy-bcubed.jar',
                         test_data_dir + '/keys/gold/all.singlesense.key', 'finalResult.out'])
result2 = subprocess.check_output(
                        ['java', '-jar', test_data_dir  + '/scoring/fuzzy-nmi.jar',
                         test_data_dir + '/keys/gold/all.singlesense.key', 'finalResult.out'])

bCubed = find_performance_string(result1.decode('utf-8'))
nmi = find_performance_string(result2.decode('utf-8'))
hm = calculateHarmonicMean(bCubed, nmi)

print()
print('Final result for ' + method + ':')
print('Final res for f1 socre for b_cubed ' + str(bCubed))
print('Final res for nmi socre for nmi ' + str(nmi))
print('harmonic mean: ' + str(hm))
