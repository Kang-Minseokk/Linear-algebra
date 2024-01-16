from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot

seed(1)
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)

pyplot.scatter(data1, data2)
pyplot.show()


# 아래에서 사용할 방법은 비모수적인 방법으로 상관계수를 구하는 방법이다.
# 상관계수는 두 변수간의 관계를 나타내는 값으로 대체로 -1에서 1 사이의 값을 가진다.
# 이 방법을 사용하면 정규분포를 따르지 않거나 분포를 모르는 경우에도 상관계수를 구할 수 있다.
from numpy.random import rand
from numpy.random import seed
from scipy.stats import spearmanr

seed(1)

data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)

coef, p = spearmanr(data1, data2)
print('Spearmans correlation coefficient: %.3f' % coef)

alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)


from numpy.random import rand
from numpy.random import seed
from scipy.stats import kendalltau

seed(1)

data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)

coef, p = kendalltau(data1, data2)
print('Kendall correlation coefficient: %.3f' % coef)

alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)

# 종합적으로 정리해보면 랭크 상관계수를 통해서 비모수적인 방법으로 두 변수간의 관계를 알아낼 수 있다.
# 위에서 공부한 내용은 spearman's 방법과 kendall's 방법이 있다.
# 두 방법 모두 상관계수와 p값을 구할 수 있고
# spearman's 는 선형적인 관계에서 많이 사용하고 kendall's 는 비선형적인 관계에서 사용한다.
