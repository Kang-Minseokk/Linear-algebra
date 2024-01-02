# from numpy.random import randn
# from numpy.random import seed
# from scipy.stats import pearsonr
#
# seed(1)
#
# data1 = 10 * randn(10000) + 50
# data2 = data1 + (10 * randn(10000) + 50)
#
# corr, _ = pearsonr(data1, data2)
# print('Pearsonr correlation: %.3f' % corr)
#

# Calculate Difference Effect Size
from numpy.random import randn
from numpy.random import seed
from numpy import var, sqrt, mean

def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = var(d1, ddof=1), var(d2, ddof=2)
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = mean(d1), mean(d2)

    return (u1 - u2) / s

seed(1)
data1 = 10 * randn(10000) + 60
data2 = 10 * randn(10000) + 55
d = cohend(data1, data2)

print('Cohens d: %.3f' % d)
