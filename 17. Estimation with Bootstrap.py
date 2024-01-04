from sklearn.utils import resample

data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

boot = resample(data, replace=True, n_samples=4, random_state=1)
print('Bootstrap Sample: %s' % boot)

oob = [x for x in data if x not in boot]
print('OOB Sample: %s' % oob)


# 샘플링 로직을 직접 구현해보자.
# 랜덤 시드가 동일한데, 값이 다르게 나온다. 원인을 아직 찾지 못하고 있다.

import random
from random import seed

def my_sampling(data, replace, n_samples, random_state):
    seed(random_state)
    result = []
    if replace:
        for i in range(n_samples):
            sample = random.choice(data)
            result.append(sample)
    else :
        for i in range(n_samples):
            sample = random.choice(data)
            result.append(sample)
            data.remove(sample)
    print(result)

data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
my_sampling(data=data, replace=True, n_samples=4, random_state=1)
