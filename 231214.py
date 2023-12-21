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

