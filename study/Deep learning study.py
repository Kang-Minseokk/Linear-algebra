from keras.models import Sequential
from keras.layers import Dense, Activation

# Sequential 클래스로 선형으로 레이어를 쌓는 신경망 객체 생성
model = Sequential()

# input 레이어에는 input_dim을 사용해서 입력 뉴런의 개수를 명시해야 합니다.
model.add(Dense(5, input_dim=2, activarion='relu'))
# hidden 레이어이며, 출력 뉴런은 1개입니다. 그리고 은닉층의 뉴런은 5개입니다.
model.add(Dense(1, activation='sigmoid'))

# 만약 활성화 함수를 추가하고 싶다면?
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Keras 함수형 API를 사용하면 입력층과 출력층을 따로 클래스로 정의할 수 있다.
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(2,))
hidden = Dense(2)(visible)

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(2,))
hidden = Dense(2)(visible)
model = Model(inputs=visible, outputs=hidden)


# 지금까지 Keras를 사용하여 모델을 어떻게 생성하는지 알아보았습니다.
# 이제부터는 모델 만드는 방식을 활용하여 간단한 신경망을 만들어 봅시다.

# 먼저 간단한 다층 퍼셉트론 모델을 만들어봅니다.
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
model.summary()
plot_model(model, to_file='plot_model_file/multilayer_perceptron_graph.png')


# Convolution Neural Network 모델을 만들어봅니다.
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
visible = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
hidden1 = Dense(10, activation='sigmoid')(pool2)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
model.summary()
plot_model(model, to_file='plot_model_file/convolutional_neural_network.png')


# 이번에는 순환 신경망 모델을 만들어봅니다.
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
visible = Input(shape=(100, 1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)
model.summary()
plot_model(model, to_file='plot_model_file/recurrent_neural_network.png')


