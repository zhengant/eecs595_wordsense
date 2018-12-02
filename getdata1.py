from xml.dom import minidom
import os

# root_dir = 'Data'
# corpus = ""
# for subdir, dirs, files in os.walk(root_dir):
#     for file in files:
#         print(file)
#         file_name = os.path.join(subdir, file)
#         if not str(file_name) == "Data/.DS_Store":
#             doc = minidom.parse(file_name)
#             items = doc.getElementsByTagName('s')
#             for item in items:
#                 text = ""
#                 pos = ""
#                 ws = ""
#                 sub_items = item.childNodes
#                 for sub in sub_items:
#                     if not sub.attributes == None:
#                         word_sense = -1
#                         part_of_speech = ""
#                         word = sub.firstChild.data
#                         if 'pos' in sub.attributes:
#                             part_of_speech = sub.attributes['pos'].value
#                             if 'wnsn' in sub.attributes:
#                                 word_sense = sub.attributes['wnsn'].value
#                             else:
#                                 word_sense = "-1"
#                         else:
#                             part_of_speech = "PUNC"
#                         text += (str(word) + '\t')
#                         pos += (str(part_of_speech) + '\t')
#                         ws += (str(word_sense) + '\t')
#                 corpus += (text + '\n' + pos + '\n' + ws + '\n')
# f = open("dataset.csv", "w")
# f.write(corpus)

path = "Datasets"
try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

root_dir = 'xml-format'
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_name = os.path.join(subdir, file)
        if not str(file_name) == "xml-format/.DS_Store":
            lemma = ""
            corpus = ("id" + "\t" + "lemma" + "\t" + "partOfSpeech" + "\t" + "token" + "\t" + "context" + "\n")
            doc = minidom.parse(file_name)
            items = doc.getElementsByTagName('instances')
            for item in items:
                lemma = item.attributes['lemma'].value
                sub_items = item.getElementsByTagName('instance')
                for sub in sub_items:
                    corpus += (sub.attributes['id'].value + "\t" + sub.attributes['lemma'].value + "\t" + sub.attributes['partOfSpeech'].value + "\t" + sub.attributes['token'].value + "\t" + sub.firstChild.data + "\n")
            outfile = ("Datasets/" + str(lemma) + ".tsv")
            f = open(outfile, "w")
            f.write(corpus)
            f.close()

# f = open("Datasets/add.csv", "w")
# f.write("hello")
# f.close()
