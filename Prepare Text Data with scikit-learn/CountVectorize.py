# CounterVectorizer 클래스를 사용하여 단어의 출현빈도를 기반으로 벡터화한다.
from sklearn.feature_extraction.text import CountVectorizer

text = ["The quick brown fox jumped over the lazy dog"]
vectorizer = CountVectorizer()
vectorizer.fit(text)

print(vectorizer.vocabulary_)

vector = vectorizer.transform(text)
print(vector.shape)
print(type(vector))
print(vector.toarray())


# TfidfVectorizer 클래스를 사용하여 단어의 중요도를 기반으로 벡터화한다.
from sklearn.feature_extraction.text import TfidfVectorizer
text = ["The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"]
vectorizer = TfidfVectorizer()
vectorizer.fit(text)

print(vectorizer.vocabulary_)
print(vectorizer.idf_)

vector = vectorizer.transform([text[0]])
print(vector.shape)
print(vector.toarray())


# HashingVectorizer 클래스는 어휘를 학습하지 않으며 고정된 크기의 벡터를 생성합니다.
# 메모리 효율적이어서 대용량의 텍스트 데이터를 다룰 수 있습니다.
# 동일한 해시값을 갖게 될 수 있습니다. 이 단점을 감수하고 사용한다고 합니다.

from sklearn.feature_extraction.text import HashingVectorizer
text = ["The quick brown fox jumped over the lazy dog."]
vectorizer = HashingVectorizer(n_features   =20)
vector = vectorizer.transform(text)

print(vector.shape)
print(vector.toarray())

