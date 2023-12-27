from numpy import array
from numpy.linalg import svd
from numpy import zeros
from numpy import diag

A = array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]])
print(A)

U, s, V = svd(A)
d = 1.0 / s
D = zeros(A.shape)

D[:A.shape[1], :A.shape[1]] = diag(d)
B = V.T.dot(D.T).dot(U.T)
print(B)

from numpy import array
from numpy import mean

M = array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]])
print(M)

col_mean = mean(M, axis=0)
row_mean = mean(M, axis=1)

print(col_mean)
print(row_mean)

from numpy import array
from numpy import var

v = array([1, 2, 3, 4, 5, 6])
print(v)

result = var(v, ddof=1)
print(result)

from numpy import array
from numpy import var

M = array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]])
print(M)

col_var = var(M, ddof=1, axis=0)
row_var = var(M, ddof=1, axis=1)

print(col_var)
print(row_var)

from numpy import array
from numpy import std

M = array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]])
print(M)

col_std = std(M, axis=0, ddof=1)
row_std = std(M, axis=1, ddof=1)
print(col_std)
print(row_std)

from numpy import array
from numpy import cov
x = array([1,2,3,4,5,6,7,8,9])
print(x)

y= array([9,8,7,6,5,4,3,2,1])
print(y)

Sigma = cov(x, y)[0, 1]
print(Sigma)

from numpy import array
from numpy import corrcoef

x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x)

y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(y)

corr = corrcoef(x, y)[0, 1]
print(corr)

from numpy import array, mean
from numpy import cov

X = array([
    [1, 5, 8],
    [3, 5, 11],
    [2, 4, 9],
    [3, 6, 10],
    [1, 5, 10]])
print(X)
print(mean(X))

Sigma = cov(X.T)
print(Sigma)

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

A = array([
    [1, 2],
    [3, 4],
    [5, 6]])
print(A)

M = mean(A.T, axis=1)
C = A - M
V = cov(C.T)
values, vectors = eig(V)
print(vectors)
print(values)
P = vectors.T.dot(C.T)
print(P.T)


from numpy import array
from sklearn.decomposition import PCA

A = array([
    [1, 2],
    [3, 4],
    [5, 6]])
print(A)

pca = PCA(2)
pca.fit(A)
print(pca.components_)
print(pca.explained_variance_)
B = pca.transform(A)
print(B)


from numpy import array
from matplotlib import pyplot

data = array([
    [0.05, 0.12],
    [0.18, 0.22],
    [0.31, 0.35],
    [0.42, 0.38],
    [0.5, 0.49]])
print(data)

X, y = data[:, 0], data[:, 1]
X = X.reshape((len(X), 1))
pyplot.scatter(X, y)
pyplot.show()


from numpy import array
from numpy.linalg import inv
from matplotlib import pyplot

data = array([
    [0.05, 0.12],
    [0.18, 0.22],
    [0.31, 0.35],
    [0.42, 0.38],
    [0.5, 0.49]])

X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
b = inv(X.T.dot(X)).dot(X.T).dot(y)
print(b)

yhat = X.dot(b)
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()



from numpy import sin, arange
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

seed(1)

x = randn(1000)
pyplot.hist(x, bins=100)
pyplot.show()

from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
seed(1)

x = [randn(1000), 5 * randn(1000), 10 * randn(1000)]
pyplot.boxplot(x)
pyplot.show()

from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
seed(1)
x = 20 * randn(1000) + 100
y = x + (10 * randn(1000) + 50)
pyplot.scatter(x, y)
pyplot.show()


from random import seed
from random import random

seed(1)

print(random(), random(), random())

seed(1)
print(random(), random(), random())

from random import seed
from random import random
seed(1)

for _ in range(10):
    value = random()
    print(value)

from random import seed
from random import randint

seed(1)

for _ in range(10):
    value = randint(0,10)
    print(value)

from random import seed
from random import gauss

seed(1)
for _ in range(10):
    value = gauss(0, 1)
    print(value)

from numpy.random import seed
from numpy.random import rand
seed(1)
print(rand(3))
seed(1)
print(rand(3))

from numpy.random import seed
from numpy.random import rand

seed(1)
values = rand(10)
print(values)

from numpy.random import seed
from numpy.random import randint

seed(1)
values = randint(0,10,20)
print(values)

from numpy.random import seed
from numpy.random import randn

seed(1)
values = randn(10)
print(values)

from numpy.random import seed
from numpy.random import shuffle

seed(1)
sequence = [i for i in range(20)]
print(sequence)
shuffle(sequence)
print(sequence)

from numpy import arange
from matplotlib import pyplot
from scipy.stats import norm

xaxis = arange(30, 70, 1)
yaxis = norm.pdf(xaxis, 50, 5)
pyplot.plot(xaxis, yaxis)


from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import array
from matplotlib import pyplot

seed(1)
sizes = list()
for x in range(10, 20000, 200):
    sizes.append(x)
means = [mean(5 * randn(size) + 50) for size in sizes]
pyplot.scatter(sizes, array(means)-50)
pyplot.show()

from numpy.random import seed
from numpy.random import randint
from numpy import mean
from matplotlib import pyplot

seed(1)

means = [mean(randint(1, 7, 50)) for _ in range(1000)]
pyplot.hist(means)
pyplot.show()
