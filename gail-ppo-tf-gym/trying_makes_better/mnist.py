# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
INPUT_NODE = 784  # 输入层节点数 → 图片像素
OUTPUT_NODE = 10  # 输出层节点数 → 0~9 这 10个数字

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数，这里使用只有一个隐藏层的网络结构作为样例
BATCH_SIZE = 100  # 一个训练 batch 中的训练数据个数。数字越小，训练过程越接近随机梯度下降，反之接近梯度下降（all data）
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
# 定义了一个使用 ReLUctant激活函数的三层全连接神经网络，通过加入隐藏层实现了多层网络结构，
# 通过 ReLUctant激活函数实现了去线性化。
# 函数也支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:  # 如果没有提供滑动平均类时，直接使用参数当前的取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)  # 计算隐藏层的前向传播结果
        return tf.matmul(layer1, weights2) + biases2
    else:

        # 首先使用 avg_class.average 函数计算得出变量的滑动平均值，然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为 None，所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里制定这个变量为不可训练的变量（trainable = False）
    # 在使用 TensorFlow 训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如 global_step）就不需要了
    # tf.trainable_variables 返回的就是图上集合 GraphKeys.TRAINABLE_VARIABLES 中的元素。
    # 这个集合的元素就是所有没有制定 trainable = False 的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算滑动平均之后的前向传播结果。滑动平均不会改变变量本身的取值，而是会维护一个影子变量来记录其滑动平均值。
    # 所以当需要使用这个滑动平均值时，需要明确调用 average 函数
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # y是不包括 softmax 层的前向传播结果， 第二个参数来得到正确答案所对应的类别编号（返回最大值的索引号，axis=1表示第二个维度）
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)

    # 计算在当前 batch 中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算 L2 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失。 一般只计算神经网络边上权重的正则化损失而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础的学习率，更新变量时使用
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 遍历完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY)  # 学习率衰减速度

    # 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数。这里的损失函数包含了交叉熵损失和 L2 正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #   在训练神经网络模型时，每遍历一边数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值。
    # 为了一次完成多个操作， 有 tf.control_dependencies 和 tf.group 两种机制。下面两行程序和
    # train_op = tf.group(train_step, variables_averages_op) 是等价的
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

        # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。
    # tf.argmax(average_y, 1)计算每一个样例的预测答案。 其中 average_y 是一个 batch_size * 10的二维数组，每一行表示一个样例的前向传播结果。
    # tf.argmax 的第二个参数“1” 表示选取最大值的操作尽在第一个唯独中进行，也就是说，只在每一行选取最大值对应的下标。于是得到的结果是一个长度为 batch的一维数组，
    # 这个一维数组中的值就表示了每一个样例对应的数字识别结果。
    # tf.equl 判断两个张量的每一维是否相等，如果相等返回 True，否则返回 False
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 将布尔型数值转换为实数型，然后计算平均值。平均值就是这个模型在这一组数据上的正确率。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 准备验证数据。一般神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 准备测试数据。 在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        validation_score = []
        test_score = []

        # 迭代地训练神经网络
        for i in range(0, TRAINING_STEPS):

            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                validation_score.append(validate_acc)
                test_score.append(test_acc)
                print("在 %d 次迭代后，验证数据集的正确率为 : %g , 测试数据集的正确率为 : %g" % (i, validate_acc, test_acc))

            # 产生这一轮使用的一个 batch 的训练数据，并运行训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("在 %d 次迭代后，测试数据集的正确率为 : %g" % (i, test_acc))

        x = range(0, TRAINING_STEPS, 1000)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, validation_score, label='模型在验证数据集上的正确率')
        ax.plot(x, test_score, label='模型在测试数据集上的正确率')

        ax.legend(loc='best')
        ax.set_ylim(0.97, 0.99)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('正确率')
        ax.set_title("验证数据集与测试数据集正确率对比")
        plt.show()


# 主程序入口
def main(argv=None):
    # 声明处理 MNIST 数据集的类， 这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


# TensorFLow 提供的一个主程序入口， tf.app.run 会调用上面定义的 main 函数
if __name__ == '__main__':
    main(argv=None)
    # tf.app.run()
