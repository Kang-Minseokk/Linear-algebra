# 아래의 코드는 최소 및 최대값과 1사분위수, 2사분위수(중앙값), 3사분위수를 구하는 코드입니다.
# 데이터 샘플이 정규 분포를 따르지 않거나 알지 못하는 경우 아래에서 구하는 5가지의 수를 구한다.
# 정규분포를 따르는 데이터 샘플에 대해서도 5가지 수를 구하고 하면 도움이 된다.

from numpy import percentile
from numpy.random import seed
from numpy.random import rand

seed(1)

data = rand(1000)

quartiles = percentile(data, [25, 50, 75])
data_min, data_max = data.min(), data.max()

print("Min: %.3f" % data_min)
print("Q1: %.3f" % quartiles[0])
print("Median: %.3f" % quartiles[1])
print("Q3: %.3f" % quartiles[2])
print("Max: %.3f" % data_max)
