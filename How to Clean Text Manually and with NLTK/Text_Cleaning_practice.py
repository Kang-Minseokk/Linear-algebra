# 공백을 기준으로 텍스트 내의 단어들을 리스트화 시키기


filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

words = text.split()
print(words[:100])

# 특정 단어 선택하기
import re
import string
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

words = re.split(r'\W+', text)
print(words[:100])


import string
import re
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
words = text.split()
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
stripped = [re_punc.sub('', w) for w in words]
print(stripped[:100])

filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
words = text.split()
words = [word.lower() for word in words]
print(words[:100])

import nltk
nltk.download()
# 이제부터는 NLTK라이브러리를 사용해서 텍스트 데이터 정제를 하려고 한다.
from nltk import sent_tokenize
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
sentences = sent_tokenize(text)
print(sentences[0])

from nltk.tokenize import word_tokenize
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
tokens = word_tokenize(text)
words = [word for word in tokens if word.isalpha()]
print(words[:100])


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)


import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# 단어를 단위로 토큰화한다.
# 이때 콤마와 같은 특수문자(?)도 하나의 단어로 취급받는다.
tokens = word_tokenize(text)
tokens = [w.lower() for w in tokens]

# 정규 표현식을 사용하여 특수문자를 제거한다.
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
stripped = [re_punc.sub('', w) for w in tokens]

# 구두점(punctuation)을 제거해주었기 때문에 해당 자리에는 빈 문자열만이 있다.
# 빈 문자열을 제거해주자.
words = [word for word in stripped if word.isalpha()]

# 흔하게 등장하는 a, the, is와 같은 stop words를 제거한다.
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words[:100])


from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

tokens = word_tokenize(text)
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])




