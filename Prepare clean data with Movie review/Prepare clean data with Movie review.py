# from os import listdir
#
# # 주어진 디렉터리 아래에 있는 파일을 리스트로 반환하는 listdir() 메서드를 사용해봅니다.
# def load_doc(filename):
#     file = open(filename, 'r')
#     text = file.read()
#     file.close()
#     return text
#
# directory = 'txt_sentoken/neg'
#
# for filename in listdir(directory):
#     if not filename.endswith(".txt"):
#         next
#     path = directory + '/' + filename
#
#     doc = load_doc(path)
#     print('Loaded %s' % filename)


# # 단어를 띄어쓰기 기준으로 나누어 토큰화하기
# def load_doc(filename):
#     file = open(filename, 'r')
#     text = file.read()
#     file.close()
#     return text
#
# filename = 'txt_sentoken/neg/cv000_29416.txt'
# text = load_doc(filename)
#
# tokens = text.split()
# print(tokens)
# # 이렇게 띄어쓰기 기준으로 토큰화를 하게되면 방점도 유지되고, stop words도 유지되기에 해결할 방법이 필요하다.



import string
import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % (re.escape(string.punctuation)))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def process_docs(directory, vocab):
    for filename in listdir(directory):
        if not filename.endswith(".txt"):
            next
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)

vocab = Counter()
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
print(len(vocab))
print(vocab.most_common(50))





