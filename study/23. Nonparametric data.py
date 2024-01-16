from numpy.random import rand
from numpy.random import seed
from scipy.stats import rankdata

seed(1)

data = rand(1000)

print(data[:10])
ranked = rankdata(data)
print(ranked[:10])

arr = [1, 4, 6, 34, 45, 43, 31]
ranked = rankdata(arr)
print(ranked)