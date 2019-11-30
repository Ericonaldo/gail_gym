import tensorflow as tf
import numpy as np
import argparse

def argparser():
    parser = argparse.ArgumentParser() #创建ArgumentParser()对象
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/energy')
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    return parser.parse_args()



class Energy_net:
    def __init__(self, name: str, env:str):
        '''
        :param name: string
        :param env: gym env
        '''

        self.training_iterations = 1000
        self.sigma = 0.1
        if env == 'CartPole-v0':
            sapairs_space = 5
        else:
            print("Environment non-ex")
        with tf.variable_scope(name):
            self.sapairs = tf.placeholder(dtype=tf.float32, shape=[None, sapairs_space], name='sapairs') #None是留给batchsize的空间
            self.noise_sapairs = tf.placeholder(dtype=tf.float32, shape=[None, sapairs_space], name='noise_sapairs')
            with tf.variable_scope('energy_net'):
                layer_1 = tf.layers.dense(inputs=self.noise_sapairs, units=200, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=200, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=200, activation=tf.tanh)
                self.energy = tf.layers.dense(inputs=layer_3, units=1, activation=tf.tanh)

            self.scope = tf.get_variable_scope().name

            E_y_gradient = tf.gradients(self.energy, self.noise_sapairs)
            SigmaSquare_E_y_gradient = tf.multiply(self.sigma**2, E_y_gradient)
            self.loss = tf.reduce_mean(tf.square(self.sapairs - self.noise_sapairs + SigmaSquare_E_y_gradient))        #0.01为方差，也就是sigma平方

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)



    def get_energy(self, sa):
        return tf.get_default_session().run(self.energy, feed_dict={self.noise_sapairs: sa})

    def loss(self, sapairs, noise_sapairs):
        return tf.get_default_session().run(self.loss, feed_dict={self.sapairs: sapairs,
                                                                  self.noise_sapairs: noise_sapairs})
    def train(self, sapairs, noise_sapairs):
        # print("loss: ", tf.get_default_session().run(self.loss, feed_dict={self.sapairs: sapairs,
        #                                                           self.noise_sapairs: noise_sapairs}))
        return tf.get_default_session().run([self.loss,self.train_op], feed_dict={self.sapairs: sapairs,
                                                                      self.noise_sapairs: noise_sapairs})
