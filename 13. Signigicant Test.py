# from numpy.random import seed
# from numpy.random import randn
# from numpy import mean
# from numpy import std
#
# seed(1)
#
# data1 = 5 * randn(100) + 50
# data2 = 5 * randn(100) + 51
# print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
# print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))


# # 일반적인 t-test
# from numpy.random import seed
# from numpy.random import randn
# from scipy.stats import ttest_ind
# seed(1)
#
# data1 = 5 * randn(100) + 50
# data2 = 5 * randn(100) + 51
#
# stat, p = ttest_ind(data1, data2)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
#
# alpha = 0.05
# if p >= alpha :
#     print("Same distribution (fail to reject H0)")
# else :
#     print("Different distribution (reject H0)")


# # paired t-test
# from numpy.random import seed
# from numpy.random import randn
# from scipy.stats import ttest_rel
#
# seed(1)
# data1 = 5 * randn(100) + 50
# data2 = 5 * randn(100) + 51
#
# stat, p = ttest_rel(data1, data2)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
#
# alpha = 0.05
# if p > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')

# ANOVA test
from numpy.random import randn
from numpy.random import seed
from scipy.stats import f_oneway

seed(1)

data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 50
data3 = 5 * randn(100) + 52

stat, p = f_oneway(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# repeated measures ANOVA Test는 아직 Scipy에 존재하지 않기에 예제가 없습니다.
