# 어떤 식으로 유사 정규분포 또는 정규분포가 아닌 분포가 있는지 확인한다.

from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

seed(1)

data = 50 * randn(100) + 100

pyplot.hist(data)
pyplot.show()


# 데이터의 해상도 (Resolution)도 중요합니다.
# 해상도를 많이 낮추니 데이터가 잘 표현되지 않는다는것을 알 수 있다.
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

seed(1)

data = randn(100)

data = data.round(0)

pyplot.hist(data)
pyplot.show()

from numpy.random import randn
from numpy.random import seed
from numpy import zeros
from numpy import append
from matplotlib import pyplot

seed(1)
data = 5 * randn(100) + 10

data = append(data, zeros(10))
pyplot.hist(data)
pyplot.show()

# 꼬리가 있는 분포.
from numpy.random import seed
from numpy.random import randn
from numpy.random import rand
from numpy import append
from matplotlib import pyplot

seed(1)
data = 5 * randn(100) + 10
tail = 10 + (rand(50) * 100)
data = append(data, tail)
pyplot.hist(data)
pyplot.show()


from numpy.random import seed
from numpy.random import randn
from numpy.random import rand
from numpy import append
from matplotlib import pyplot

seed(1)

data = 5 * randn(100) + 10
tail = 10 + (rand(10) * 100)

data = append(data, tail)
data = [x for x in data if x < 25]
pyplot.hist(data)
pyplot.show()
