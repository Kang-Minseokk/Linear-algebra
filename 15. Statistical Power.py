# p-값은 귀무가설이 참일 때 해당 결과를 관측할 확률을 나타낸다.
# p-값이 커지면 해당 사건이 우연히 일어났을 확률이 커지고 통계적으로 유의미하지 않다고 판단할 수 있다.

# Sample Size를 추정하자.
from statsmodels.stats.power import TTestIndPower

effect = 0.8
alpha = 0.05
power = 0.8
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)


# 표본의 크기와 effect size를 다양화했을 때 변화하는 검증력을 그래프에 나타내자.
from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower

effect_sizes = array([0.2, 0.5, 0.8])
sample_sizes = array(range(5, 100))

analysis = TTestIndPower()
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()
