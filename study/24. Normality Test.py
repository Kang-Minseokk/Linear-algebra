from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
from matplotlib import pyplot

seed(1)

data = 5 * randn(100) + 50

print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))

# 히스토그램으로 데이터의 정규 분포를 따르는지를 확인할 수 있다.
pyplot.hist(data)
pyplot.show()


from numpy.random import seed
from numpy.random import randn
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

seed(1)
data = 5 * randn(100) + 50
qqplot(data, line='s')
pyplot.show()


# Shapiro-Wilk Test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
seed(1)

data = 5 * randn(100) + 50

stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else :
    print('Sample does not look Gaussian (reject H0)')


# D'Agostino's K2 Test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import normaltest

seed(1)

data = 5 * randn(100) + 50
stat, p = normaltest(data)

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


from numpy.random import seed
from numpy.random import randn
from scipy.stats import anderson

seed(1)

data = 5 * randn(100) + 50

result = anderson(data)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

