import argparse
import gym
import numpy as np
import tensorflow as tf
import random
from network_models.energy_net import Energy_net

def argparser():
    parser = argparse.ArgumentParser() #创建ArgumentParser()对象
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/energy')
    parser.add_argument('--iteration', default=int(1e6), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--epoch_num', default=500, type=int)
    return parser.parse_args()

def main(args):
    # env = gym.make('CartPole-v0')
    Energy = Energy_net('energy', 'CartPole-v0')
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    sapairs = np.genfromtxt('training_data/sapairs.csv')
    noise_sapairs = np.genfromtxt('training_data/noise_sapairs.csv')
    print('sapairs X we get is:' , sapairs)
    print('noise sapairs Y we get is: ', noise_sapairs)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph) #制定一个文件用来保存图
        sess.run(tf.global_variables_initializer())

        # train

        inp = [sapairs, noise_sapairs]

        for iteration in range(args.iteration): #训练外循环次数
            loss_for_this_training_iteration = []
            for epoch in range(args.epoch_num):  #训练内循环次数
                # select sample indices in [low, high]
                sample_indices = np.random.randint(low=0, high=noise_sapairs.shape[0], size=args.minibatch_size)
                # print("Sample: ", sample_indices)
                sample_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                loss =   Energy.train(sapairs = sample_inp[0], noise_sapairs=sample_inp[1])[0]
                loss_for_this_training_iteration.append(loss)
                # np.append(loss_for_this_training_iteration, loss)
                # print("sample_inp :", sample_inp)
                # print("sample_inp[0]", sample_inp[0])
                # print("sample_inp[1]", sample_inp[1])
            print("training outer iteration", iteration)
            print("Mean loss=",np.mean(np.array(loss_for_this_training_iteration)))

if __name__ == '__main__':
    args = argparser()
    main(args)