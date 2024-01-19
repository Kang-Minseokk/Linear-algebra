from keras.preprocessing.text import text_to_word_sequence

text = 'The quick brown fox jumped over the lazy dog.'
result = text_to_word_sequence(text)
print(result)


from keras.preprocessing.text import text_to_word_sequence

text = "The quick brown fox jumped over the lazy dog."
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence

text = 'The quick brown fox jumped over the lazy dog.'

words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

result = one_hot(text, round(vocab_size*1.3))
print(result)




# 해싱 트릭 함수를 사용해본다.
# 해싱 트릭 함수에는 단어를 분리해주는 기능이 존재하지 않기 때문에 직접 text_to_word_sequence로 한다.
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence

text = 'The quick browm fox jumped over the lazy dog.'
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)

from keras.preprocessing.text import Tokenizer
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!']

t = Tokenizer()
t.fit_on_texts(docs)

print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)


