import argparse
import gym
import numpy as np
import tensorflow as tf
import random

# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


states = np.genfromtxt("../trajectory/observations.csv")
actions = np.genfromtxt("../trajectory/actions.csv")[0:]
print(states)
print("We have got", len(states), "states")
print(actions)
print("We have got", len(actions), "actions")


#有saNumber个state-action pair
saNumber = len(actions)
#一个state-action pair有saShape个元素
saShape = len(states[0]) + 1  #因为这个地方action space = 1
print("saNumber = ", saNumber)
print("saShape = ", saShape)


# 搞成state-action pair
sapairs = []
for i in range(0, len(states)):
    sapairs.append([states[i][0], states[i][0], states[i][0], states[i][0], actions[i]])
print(sapairs)
print("We have got", len(sapairs), "state-action pairs")

# gauss noise
# 定义 gauss noise 的均值和方差
mu, sigma = 0, 0.1

# 一维guass
sampleNo = saShape * saNumber  # 采样sampleNo个gauss noise
noise = np.random.normal(mu, sigma, sampleNo)
print("noise:", noise)
print("We have got", len(noise), "noises")


#转成np array 便于加noise
sapairs = np.array(sapairs)
print("numpy array version sapairs: " , sapairs)
noise = np.array(noise)
print("numpy array version noise:", noise)
noise_sapairs = np.reshape(sapairs, newshape=[saNumber*saShape])+noise
print("After adding noise, noise sapairs = " , noise_sapairs)
noise_sapairs = np.reshape(noise_sapairs, newshape=[saNumber,saShape])
print("After adding reshaping, noise sapairs = " , noise_sapairs)

noise = np.reshape(noise, newshape=[saNumber,saShape])



open_file_and_save('training_data/sapairs.csv', sapairs)
open_file_and_save('training_data/noise_sapairs.csv', noise_sapairs)
open_file_and_save('training_data/noise.csv', noise)