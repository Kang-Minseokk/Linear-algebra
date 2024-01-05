from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import sqrt
from scipy.stats import chi2
from scipy.stats import norm

seed(1)
# 데이터 샘플을 정규 분포에 따라 생성한다.
data = 5 * randn(100) + 50

# 자유도를 구한다.
n = len(data)
dof = n - 1

# 임계값 구하기 (주어진 확률에 해당하는 표준 확률 분포의 백분위 수 구하기)
prop = 0.95
prop_inv = (1.0 - prop) / 2.0
gauss_critical = norm.ppf(prop_inv)
print('Gaussian critical value: %.3f (coverage=%d%%)' % (gauss_critical, prop*100))


prob = 0.99
prop_inv = 1.0 - prob
chi_critical = chi2.ppf(prop_inv, dof)
print('Chi-Squared critical value: %.3f (prob=%d%%, dof=%d)' % (chi_critical, prob * 100, dof))

interval = sqrt((dof * (1 + (1/n)) * gauss_critical**2) / chi_critical)
print('Tolerance Interval: %.3f' % interval)

data_mean = mean(data)
lower, upper = data_mean-interval, data_mean+interval
print('%.3f to %.2f covers %d%% of data with a confidence of %d%%' % (lower, upper, prop*100, prob*100))


