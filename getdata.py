from xml.dom import minidom
import os

root_dir = 'Data'
corpus = ""
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        print(file)
        file_name = os.path.join(subdir, file)
        if not str(file_name) == "Data/.DS_Store":
            doc = minidom.parse(file_name)
            items = doc.getElementsByTagName('s')
            for item in items:
                text = ""
                pos = ""
                ws = ""
                sub_items = item.childNodes
                for sub in sub_items:
                    if not sub.attributes == None:
                        word_sense = -1
                        part_of_speech = ""
                        word = sub.firstChild.data
                        if 'pos' in sub.attributes:
                            part_of_speech = sub.attributes['pos'].value
                            if 'wnsn' in sub.attributes:
                                word_sense = sub.attributes['wnsn'].value
                            else:
                                word_sense = "-1"
                        else:
                            part_of_speech = "PUNC"
                        text += (str(word) + '\t')
                        pos += (str(part_of_speech) + '\t')
                        ws += (str(word_sense) + '\t')
                corpus += (text + '\n' + pos + '\n' + ws + '\n')
f = open("dataset.csv", "w")
f.write(corpus)
