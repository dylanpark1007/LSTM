import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


en = []
with open('./data/WMT14_en.txt','r', encoding='utf-8') as f:
    idx = 0
    line = f.readline()
    en.append(clean_str(line))
    while line != '':
        line = f.readline()
        en.append(clean_str(line))
        idx += 1
        if idx % 100000 == 0:
            print(idx)


de = []
with open('./data/WMT14_de.txt','r', encoding='utf-8') as f:
    idx = 0
    line = f.readline()
    de.append(clean_str(line))
    while line != '':
        line = f.readline()
        de.append(clean_str(line))
        idx += 1
        if idx % 100000 == 0:
            print(idx)

en_de = []
for idx,line in enumerate(en):
    merged = line + '.' + '\t' + de[idx] + '.' + '\n'
    en_de.append(merged)

with open('./data/de_eng.txt','w', encoding='utf-8') as f:
    f.writelines(en_de)

en_de1 = en_de[0:200000]



test2 = []
with open('./data/spa-eng.txt','r', encoding='utf-8') as f:
    line = f.readline()
    test2.append(line)
    idx = 0
    while line != None:
        line = f.readline()
        test2.append(line)
        idx += 1
        if idx == 100000:
            break