# p-값은 귀무가설이 참일 때 해당 결과를 관측할 확률을 나타낸다.
# p-값이 커지면 해당 사건이 우연히 일어났을 확률이 커지고 통계적으로 유의미하지 않다고 판단할 수 있다.

# Sample Size를 추정하자.
from statsmodels.stats.power import TTestIndPower
from numpy import array
from matplotlib import pyplot

effect = 0.8
alpha = 0.05
nobs1 = 40
power = 0.8
sample_sizes = array(range(5, 100))
effect_sizes = array([0.8])
analysis = TTestIndPower()

# 검정력을 구해보자.
result = analysis.solve_power(effect, power=None, nobs1=nobs1, ratio=1.0, alpha=alpha)
print('Power: %.3f' % result)

# 표본의 크기를 구해보자.
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)


# 표본의 크기에 따라서 어떻게 변하는지 확인을 해보자
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()


# 표본의 크기와 effect size를 다양화했을 때 변화하는 검증력을 그래프에 나타내자.
from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower

effect_sizes = array([0.2, 0.5, 0.8])
sample_sizes = array(range(5, 100))

analysis = TTestIndPower()
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()

from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower

effect_sizes = array([0.2, 0.5, 0.8])
sample_sizes = array(range(5, 100))

analysis = TTestIndPower()
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()


# 서로 다른 표준 유의수준에 대한 표본 크기에 대한 검정력 곡선을 그려보세요
# 유의 수준은 통상적으로 0.01, 0.05로 설정하고 표본의 크기와 효과 크기를 바꾸어가면서 검증률을 확인하는 것이 보편적이다.
# 그래서 유의 수준을 바꾸었을 때 검증률을 확인하려면 우리가 따로 계산을 해주어야 한다.
# 자세하기 어떠한 계산이냐면 alpha값에 따른 검증률 값을 solve_power() 메서드로 구하고 해당 값들을 배열로 만들어서 그래프를 그려준다.
# 처음에는 이러한 사실을 모르고 오류를 해결하지 못했다.
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

# Set the effect size and alpha values
effect_size = 0.5
alpha_values = [0.01, 0.05, 0.1, 0.2]  # Add more if needed

# Create a range of sample sizes
sample_sizes = range(10, 200, 10)  # Adjust the range based on your needs

# Initialize the TTestIndPower class
analysis = TTestIndPower()

# Plot power curves for different alpha values
for alpha in alpha_values:
    power_values = [analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha) for n in sample_sizes]
    plt.plot(sample_sizes, power_values, label=f'Alpha={alpha}')

# Add labels and legend
plt.xlabel('Sample Size')
plt.ylabel('Power')
plt.title('Power Curve for Different Alpha Levels')
plt.legend()

# Show the plot
plt.show()

