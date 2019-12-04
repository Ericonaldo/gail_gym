import argparse
import gym
import numpy as np
import tensorflow as tf
import random
from network_models.energy_net import Energy_net
import time
# 格式化成2016-03-20 11:45:39形式
date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# energy_training_data 用于存储训练日志文件
energy_training_data = []
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

def argparser():
    parser = argparse.ArgumentParser() #创建ArgumentParser()对象
    parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/energy/fixed_noise')
    parser.add_argument('--max_to_keep', help='number of models to save', default=1000, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/energy')
    parser.add_argument('--iteration', default=int(1e4), type=int)
    parser.add_argument('--interval', help='save interval', default=int(10), type=int)
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--epoch_num', default=500, type=int)
    parser.add_argument('--noise_sigma', default=0.1, type=float)
    parser.add_argument('--environment', )
    return parser.parse_args()


def main(args):
    print(date)
    energy_training_data.append(["Date:",date])
    energy_training_data.append(["Noise type:", "fixed noise"])
    energy_training_data.append(["Energy Training iterations:",args.iteration])
    energy_training_data.append(["Save interval:", args.interval])
    energy_training_data.append(["minibatch_size:", args.minibatch_size])
    energy_training_data.append(["epoch_num for one training iteration:", args.epoch_num])
    energy_training_data.append(["gauss noise sigma:", args.noise_sigma])

    open_file_and_save(args.logdir+'/'+'fixed_noise_Energy_'+date, energy_training_data)


    # env = gym.make('CartPole-v0')
    Energy = Energy_net('energy', 'CartPole-v0')
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    sapairs = np.genfromtxt('training_data/sapairs.csv')

    # 定义 gauss noise 的均值和方差
    mu, sigma = 0, args.noise_sigma
    # 一维guass
    saNumber = sapairs.shape[0]
    saShape = sapairs.shape[1]
    sampleNo = saNumber * saShape  # 采样sampleNo个gauss noise
    noise = np.random.normal(mu, sigma, sampleNo)
    noise_sapairs = np.reshape(sapairs, newshape=[saNumber * saShape]) + noise
    noise_sapairs = np.reshape(noise_sapairs, newshape=[saNumber, saShape])

    # noise_sapairs = np.genfromtxt('training_data/noise_sapairs.csv')
    # print("noise:" , noise[4],noise[9])
    print('sapairs X we get is:' , sapairs)
    print('noise sapairs Y we get is: ', noise_sapairs)
    print("Training iterations:", args.iteration)


    with tf.Session() as sess:
        # writer = tf.summary.FileWriter(args.logdir, sess.graph) #制定一个文件用来保存图
        sess.run(tf.global_variables_initializer())

        # train

        inp = [sapairs, noise_sapairs]

        for iteration in range(args.iteration): #训练外循环次数
            loss_for_this_training_iteration = []
            energy_training_data_for_this_iteration= []
            for epoch in range(args.epoch_num):  #训练内循环次数
                # select sample indices in [low, high]
                sample_indices = np.random.randint(low=0, high=noise_sapairs.shape[0], size=args.minibatch_size)
                # print("Sample: ", sample_indices)
                sample_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                #print(sample_inp[0])
                loss =   Energy.train(sapairs = sample_inp[0], noise_sapairs=sample_inp[1])[0]
                loss_for_this_training_iteration.append(loss)
                # np.append(loss_for_this_training_iteration, loss)
                # print("sample_inp :", sample_inp)
                # print("sample_inp[0]", sample_inp[0])
                # print("sample_inp[1]", sample_inp[1])
            print("training outer iteration", iteration)
            print("Mean loss=",np.mean(np.array(loss_for_this_training_iteration)))
            energy_training_data_for_this_iteration.append(["training outer iteration", iteration,"Mean loss=",np.mean(np.array(loss_for_this_training_iteration))])

            open_file_and_save(args.logdir+'/'+'fixed_noise_Energy_'+date, energy_training_data_for_this_iteration)


            if (iteration+1) % args.interval == 0:
                saver.save(sess, args.savedir + '/model.ckpt', global_step=iteration+1)
                open_file_and_save(args.logdir + '/' + 'fixed_noise_Energy_' + date,
                                   [["Energy model saved"]])
    print(date)


if __name__ == '__main__':
    args = argparser()
    main(args)