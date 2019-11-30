import argparse
import gym
import numpy as np
import tensorflow as tf
import random
from network_models.energy_net import Energy_net

def argparser():
    parser = argparse.ArgumentParser() #创建ArgumentParser()对象
    parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/energy/new_noise')
    parser.add_argument('--max_to_keep', help='number of models to save', default=100, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/energy')
    parser.add_argument('--iteration', default=int(1e5), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=500, type=int)
    return parser.parse_args()

def main(args):
    # env = gym.make('CartPole-v0')
    Energy = Energy_net('energy', 'CartPole-v0')
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    sapairs = np.genfromtxt('training_data/sapairs.csv')
    print(type(sapairs))
    print("sapairs shape:", sapairs.shape)
    # noise_sapairs = np.genfromtxt('training_data/noise_sapairs.csv')
    print('sapairs X we get is:' , sapairs)
    # print('noise sapairs Y we get is: ', noise_sapairs)
    print("Training iterations:", args.iteration)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph) #制定一个文件用来保存图
        sess.run(tf.global_variables_initializer())

        # train

        inp = [sapairs]

        for iteration in range(args.iteration): #训练外循环次数
            loss_for_this_training_iteration = []
            for epoch in range(args.epoch_num):  #训练内循环次数
                # select sample indices in [low, high]
                sapairs_sample_indices = np.random.randint(low=0, high=sapairs.shape[0], size=args.minibatch_size)
                # print("saparis_sample_indices: ", sapairs_sample_indices)
                sapairs_sample_inp = [np.take(a=a, indices=sapairs_sample_indices, axis=0) for a in inp]
                sapairs_sample_inp = sapairs_sample_inp[0] # 转为 [[],[],[]......[]]
                # print("sapairs_sample_inp:", sapairs_sample_inp)

                # add gauss noise

                # print("If we add 1 to sapairs_sample_inp, we get \n", sapairs_sample_inp+np.array([1,2,3,4,5]))
                #
                # print("sapairs_sample_inp.shape", sapairs_sample_inp.shape)  # shape为(32,5)

                # 定义 gauss noise 的均值和方差
                mu, sigma = 0, 0.1
                # 一维guass
                saNumber = sapairs_sample_inp.shape[0]
                saShape = sapairs_sample_inp.shape[1]
                sampleNo = saNumber * saShape  # 采样sampleNo个gauss noise
                noise = np.random.normal(mu, sigma, sampleNo)
                # print("noise:", noise)
                # print("We have got", len(noise), "noises")
                # print("numpy array version noise:", noise)
                noise_sapairs_sample_inp = np.reshape(sapairs_sample_inp, newshape=[saNumber * saShape]) + noise
                noise_sapairs_sample_inp = np.reshape(noise_sapairs_sample_inp, newshape=[saNumber, saShape])
                # print("noise_sapairs_sample_inp",noise_sapairs_sample_inp)
                loss =   Energy.train(sapairs = sapairs_sample_inp, noise_sapairs=noise_sapairs_sample_inp)[0]
                loss_for_this_training_iteration.append(loss)
                # np.append(loss_for_this_training_iteration, loss)
                # print("sample_inp :", sample_inp)
                # print("sample_inp[0]", sample_inp[0])
                # print("sample_inp[1]", sample_inp[1])
            print("training outer iteration", iteration)
            print("Mean loss=",np.mean(np.array(loss_for_this_training_iteration)))



            if (iteration+1) % args.interval == 0:
                saver.save(sess, args.savedir + '/model.ckpt', global_step=iteration+1)

if __name__ == '__main__':
    args = argparser()
    main(args)