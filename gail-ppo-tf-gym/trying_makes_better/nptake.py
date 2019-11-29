import numpy as np
data=np.genfromtxt("../trajectory/observations.csv")[0:10]
sample_indices = np.random.randint(low=0, high=data.shape[0], size=128)   #函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
print(sample_indices)