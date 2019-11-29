import numpy as np


sapairs = np.genfromtxt('../training_data/sapairs.csv')
noise_sapairs = np.genfromtxt('../training_data/noise_sapairs.csv')

sample_indices = np.random.randint(low=0, high=noise_sapairs.shape[0], size=2)
print("Sample: ", sample_indices)
print("noise_sapairs: ", noise_sapairs)


inp = [sapairs,noise_sapairs]
sample_inp = [np.take(a=a, indices=[0,1,2,3,4], axis=0) for a in inp]

print("sample_inp :", sample_inp)

print("sample_inp[0]",sample_inp[0])


