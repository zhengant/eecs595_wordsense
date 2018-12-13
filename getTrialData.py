from xml.dom import minidom
import os

path = "TrialDatasets"
try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

file_name = "semeval-2013-task-10-trial-data.xml"
doc = minidom.parse(file_name)
items = doc.getElementsByTagName('instance')
temp = ""
corpus = ""
outfile = ""
for item in items:
    lemma = item.attributes['lemma'].value
    if lemma == temp:
        corpus += (item.attributes['id'].value + "\t" + item.attributes['lemma'].value + "\t" + item.attributes[
            'partOfSpeech'].value + "\t" + item.attributes['token'].value + "\t" + item.firstChild.data + "\n")
    else:
        if lemma == "add":
            outfile = ("Datasets1/" + str(lemma) + ".tsv")
            corpus = ("id" + "\t" + "lemma" + "\t" + "partOfSpeech" + "\t" + "token" + "\t" + "context" + "\n")
            corpus += (item.attributes['id'].value + "\t" + item.attributes['lemma'].value + "\t" + item.attributes[
                'partOfSpeech'].value + "\t"
                       + item.attributes['token'].value + "\t" + item.firstChild.data + "\n")
        else:
            f = open(outfile, "w")
            f.write(corpus)
            f.close()
            outfile = ("Datasets1/" + str(lemma) + ".tsv")
            corpus = ("id" + "\t" + "lemma" + "\t" + "partOfSpeech" + "\t" + "token" + "\t" + "context" + "\n")
            corpus += (item.attributes['id'].value + "\t" + item.attributes['lemma'].value + "\t" + item.attributes['partOfSpeech'].value + "\t"
               + item.attributes['token'].value + "\t" + item.firstChild.data + "\n")
    temp = lemma
f = open(outfile, "w")
f.write(corpus)
f.close()

