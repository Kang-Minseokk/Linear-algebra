# Python Project Template

# 1. Prepare Problem
# a) Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# b) Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)


# 2. Summarize Data
# a) Descriptive statistics
# shape 출력하기
print(dataset.shape)
# 20개 데이터 순서대로 출력하기
print(dataset.head(20))
# 데이터 요약하기
print(dataset.describe())
# 데이터 분류하기
print(dataset.groupby('class').size())

# b) Data visualizations
# 박스형으로 시각화하기
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=True, sharey=False)
pyplot.show()
# 히스토그램으로 시각화하기
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
print(X_train)

# b) Test options and evaluation metric
# c) Spot Check Algorithms
models = [('LR', LogisticRegression(max_iter=1000)), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC())]
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# # d) Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# KNN이 가장 좋은 성능을 보여준다면 해당 알고리즘을 사용해서 모델을 학습시키고, 예측을 하도록 해보자.
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# 이 파일에서는 가장 기초적인 머신러닝 모델을 학습시키는 과정을 살펴보었습니다.
# 정리를 해보겠습니다.

# 데이터의 분포를 파악하고 특징을 찾아냅니다.
# 모델을 학습시키기 위한 여러가지 알고리즘을 사용합니다.
# 다양한 알고리즘으로 모델을 학습시키고 나서 각 모델의 성능을 비교합니다. (이번에는 K-Fold 교차 검증을 사용합니다.)
# 가장 좋은 성능을 가진 모델을 선택하여 직접 예측을 수행합니다.
# 예측을 수행할 때는 confusion_matrix와 정확도와 같이 예측 성능을 확인할 수 있는 지표를 함께 출력해줍니다.
