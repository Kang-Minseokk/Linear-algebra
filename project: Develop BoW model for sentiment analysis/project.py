from nltk.corpus import stopwords
import string
import re
from collections import Counter
from os import listdir


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    # 시퀀스를 가져와야 하니깐,
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def process_docs(directory, vocab):
    for filename in listdir(directory):
        if filename.startswith('cv9'):
            continue
        # 현재위치 기준으로 파일을 명시해야 하기 때문에 path를 정의한다.
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)


def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


vocab = Counter()
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
print(len(vocab))
min_occurane = 2
tokens = [k for k, c in vocab.items() if c >= min_occurane]
print(len(tokens))

save_list(tokens, 'vocab.txt')


