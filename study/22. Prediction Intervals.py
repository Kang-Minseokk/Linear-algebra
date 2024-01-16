# 차근차근 예측 구간을 구하는 과정을 설명하려고 합니다.
# 중복되는 코드가 있으니 주의하세요!


# 1. 데이터 샘플 만들기 -- 그냥 예시 데이터를 만드는 과정입니다. 이때 주의해야 할 것은 x와 y가 선형관계를 따르고 있어야 한다는 점입니다.
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot

seed(1)

x = 20 * randn(1000) + 100
y = x + (10 * randn(1000) + 50)

print('x: mean=%.3f stdv=%.3f' % (mean(x), std(x)))
print('x: mean=%.3f stdv=%.3f' % (mean(y), std(y)))

pyplot.scatter(x, y)
pyplot.show()


# 2. 위의 데이터를 바탕으로 linregress() 메서드를 통해서 예측값을 구합니다.
from numpy.random import randn
from numpy.random import seed
from scipy.stats import linregress
from matplotlib import pyplot

seed(1)

x = 20 * randn(1000) + 100
y = x + (10 * randn(1000) + 50)

b1, b0, r_value, p_value, std_err = linregress(x, y)
print('b1=%.3f, b0=%.3f' % (b1, b0))

yhat = b0 + b1 * x

pyplot.scatter(x, y)
pyplot.plot(x, yhat, color='r')

pyplot.show()

# 이제 마지막으로 예측 구간을 구해보겠습니다.
from numpy.random import randn
from numpy.random import seed
from numpy import sqrt
from numpy import sum as arraysum
from scipy.stats import linregress
from matplotlib import pyplot

seed(1)

x = 20 * randn(1000) + 100
y = x + (10 * randn(1000) + 50)

b1, b0, r_value, p_value, std_err = linregress(x, y)

yhat = b0 + b1 * x

x_in = x[0]
y_out = y[0]
yhat_out = yhat[0]

sum_errs = arraysum((y - yhat)**2)
stdev = sqrt(1/(len(y)-2) * sum_errs)
interval = 1.96 * stdev
print('Prediction Interval: %.3f' % interval)
lower, upper = y_out - interval, y_out + interval
print('95%% likelihood that the true value is between %.3f and %.3f' % (lower, upper))
print('Tru value: %.3f' % yhat_out)
pyplot.scatter(x, y)
pyplot.plot(x, yhat, color='red')
pyplot.errorbar(x_in, yhat_out, yerr=interval, color='black', fmt='o')
pyplot.show()
