# from numpy.random import seed
# from numpy.random import rand
# from scipy.stats import mannwhitneyu
#
# seed(1)
#
# data1 = 50 + (rand(100) * 10)
# data2 = 51 + (rand(100) * 10)
#
# stat, p = mannwhitneyu(data1, data2)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
#
# alpha = 0.05
# if p > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')


# # 위의 경우는 두 샘플이 서로 독립적인 경우에 사용할 수 있는 Mann-Whitney U Test라는 방법이다.
# # 아래에서 사용할 방법은 동일한 모집단에서 나온 두 샘플(쌍을 이루고 있다고 표현합니다.)인 경우에 사용하는 방법을 사용해봅니다.
# from numpy.random import seed
# from numpy.random import rand
# from scipy.stats import wilcoxon
#
# seed(1)
#
# data1 = 50 + (rand(100) * 10)
# data2 = 51 + (rand(100) * 10)
#
# stat, p = wilcoxon(data1, data2)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# alpha = 0.05
# if p > alpha:
#     print("Same distribution (fail to reject H0)")
# else:
#     print("Different distribution (reject H0)")


# # 이 방법은 ANOVA(일원 분산 분석)의 비모수 버전이라고 생각하면 쉽다.
# # 두 데이터을 비교하는 것이 아니라 여러 데이터 샘플 중에서 두개 이상의 샘플이 다른 분포를 가지고 있는지 확인할  사용한다.
# from numpy.random import seed
# from numpy.random import rand
# from scipy.stats import kruskal
#
# seed(1)
#
# data1 = 50 + (rand(100) * 10)
# data2 = 51 + (rand(100) * 10)
# data3 = 52 + (rand(100) * 10)
#
# stat, p = kruskal(data1, data2, data3)
# print('Statistics: %.3f, p: %.3f' % (stat, p))
#
# alpha = 0.05
# if p > alpha:
#     print("Same distribution (fail to reject H0)")
# else:
#     print('Different distributions (reject H0)')


# # 이 방법은 Reapeted Measures ANOVA방법의 비모수 버전이다.
# # 동일한 모집단에서 수집한 여러 데이터 중에서 두 개 이상이 동일한 분포임을 확인한다.
# from numpy.random import seed
# from numpy.random import rand
# from scipy.stats import friedmanchisquare
#
# seed(1)
#
# data1 = 50 + (10 * rand(100))
# data2 = 51 + (10 * rand(100))
# data3 = 52 + (10 * rand(100))
#
# stat, p = friedmanchisquare(data1, data2, data3)
# print('Statistics: %.3f, p-value: %.3f' % (stat, p))
#
# alpha = 0.05
# if p > alpha:
#     print("Same distributions (fail to reject H0)")
# else:
#     print("Different distributions (reject H0)")


# 이번에는 카이제곱을 사용해서 예측값과 실제 관찰값이 차이가 얼마나 발생하는지 확인한다.
# 카이제곱의 귀무가설(H0)은 예측값과 실제 관측값이 동일하다고 주장한다.
# 동일하게 된다면 통계적으로 유의하지 않게된다. 귀무 가설이 틀리다면 통계적으로 유의하기에 또 다른 분포임을 추정할 수 있다.
from scipy.stats import chi2_contingency
from scipy.stats import chi2

table = [ [10, 20, 30],
          [6, 9, 17]]
print(table)
stat, p, dof, expected = chi2_contingency(table)
print('dof=%d' % dof)
print(expected)

prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

