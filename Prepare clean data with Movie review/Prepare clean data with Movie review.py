from os import listdir

# 주어진 디렉터리 아래에 있는 파일을 리스트로 반환하는 listdir() 메서드를 사용해봅니다.
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

directory = 'txt_sentoken/neg'

for filename in listdir(directory):
    if not filename.endswith(".txt"):
        next
    path = directory + '/' + filename

    doc = load_doc(path)
    print('Loaded %s' % filename)


# 단어를 띄어쓰기 기준으로 나누어 토큰화하기
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

filename = 'txt_sentoken/neg/cv000_29416.txt'
text = load_doc(filename)

tokens = text.split()
print(tokens)
# 이렇게 띄어쓰기 기준으로 토큰화를 하게되면 방점도 유지되고, stop words도 유지되기에 해결할 방법이 필요하다.



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

# 위 코드에 대한 설명을 해보겠습니다.
# 우선 대략적인 흐름을 먼저 말씀드리자면 텍스트 데이터를 단어로 토큰화한 리스트 형식으로 변환해주고
# 해당 리스트의 요소로 존재하는 단어의 개수를 측정합니다.
# 순서를 정리해보겠습니다.
# 1. 파일을 객체로서 로드합니다.
# 2. 해당 파일의 텍스트를 읽어옵니다.
# 3. 띄어쓰기 단위로 리스트를 만들어줍니다.
# 4. 각 요소에 대해서 방점(punctuation)을 제거하고, 한글자 단어를 제거하고, stop words를 제거합니다.
# 5. Counter클래스를 사용하여 각 리스트의 단어 등장 횟수를 종합합니다.
# 6. 이 동작을 확장자가 txt인 파일에 대해서 모두 수행합니다.


# 1번 나온 단어는 예측에 큰 영향을 미치지 않습니다.
# 5번 이하로 등장한 단어에 대해서는 제거를 하도록 합니다.
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


def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


vocab = Counter()
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
print(len(vocab))
print(vocab.most_common(50))
min_occurane = 5
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))

save_list(tokens, 'vocab.txt')


import string
import re
from os import listdir
from nltk.corpus import stopwords

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
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_docs(directory, vocab):
    lines = list()
    for filename in listdir(directory):
        if not filename.endswith(".txt"):
            next
        path = directory + '/' + filename

        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
negative_lines = process_docs('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')
positive_lines = process_docs('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')


