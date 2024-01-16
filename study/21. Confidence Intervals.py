# from math import sqrt
#
# interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 50)
# print('%.3f' % interval)
#
# interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 100)
# print('%.3f' % interval)
#
# from statsmodels.stats.proportion import proportion_confint
# lower, upper = proportion_confint(88, 100, 0.05)
# print('lower=%.3f, upper=%.3f' % (lower, upper))
#


from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
from numpy import mean
from numpy import median
from numpy import percentile

seed(1)

dataset = 0.5 + rand(1000) * 0.5

scores = list()
for _ in range(100):
    indices = randint(0, 1000, 1000)
    sample = dataset[indices]
    statistic = mean(sample)
    scores.append(statistic)
print('50th percentile (media) = %.3f' % median(scores))

alpha = 5.0

lower_p = alpha / 2.0
lower = max(0.0, percentile(scores, lower_p))
print('%.1fth percentile = %.3f' % (lower_p, lower))

upper_p = (100 - alpha) + (alpha / 2.0)
upper = min(1.0, percentile(scores, upper_p))
print('%.1fth percentile = %.3f' % (upper_p, upper))

