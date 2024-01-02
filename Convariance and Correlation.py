from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot

seed(1)

data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)

print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))

pyplot.scatter(data1, data2)
pyplot.show()

# 공분산 구하기
from numpy.random import randn
from numpy.random import seed
from numpy import cov

seed(1)

data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
covariance = cov(data1, data2)
print(covariance)


# 피어슨 상관계수를 계산해보자
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr

seed(1)

data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)

corr, p = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

alpha = 0.05
if p > alpha:
    print('No correlation (fail to reject H)')
else:
    print('Some correlation (reject H0)')
