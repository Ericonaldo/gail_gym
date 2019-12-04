import gym
import numpy as np
import tensorflow as tf
import argparse
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain
from network_models.energy_net import Energy_net
from tools import kl_divergence
import matplotlib as mpl
import matplotlib.pyplot as plt


render = True
import time

# 格式化成2016-03-20 11:45:39形式
date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# energyPolicy_training_data 用于存储训练日志文件
energyPolicy_training_data = []


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

# def open_file_and_write_summary(file_path, data, line_to_replace = 28):
#     try:
#         with open(file_path+'.txt', 'r') as file:
#             lines = file.readlines()
#         # now we have an array of lines. If we want to edit the line 28...
#             for i in range (0, len(data)):
#
#
#     except FileNotFoundError:
#         print("File not exist")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='directory of model', default='trained_models')
    parser.add_argument('--alg', help='chose algorithm one of gail, ppo, bc, kl_bc, energy', default='energy')
    parser.add_argument('--noise_type', help='chose noise type for energy model(new_noise or fixed_noise)',
                        default='fixed_noise')
    parser.add_argument('--model', help='number of model to test. model.ckpt-number', default='20')
    parser.add_argument('--logdir', help='log directory', default='log/train/energy_policy')
    parser.add_argument('--iteration', default=int(2e1))
    parser.add_argument('--stochastic', action='store_false')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--sanoise', help='whether we give noise to the sapair we encounter', default=False, type=bool)
    parser.add_argument('--noise_sigma', help='The noise we add to sapair', default=0.1, type=float)
    parser.add_argument('--reward_function', help='Reward = h(energy)', default="exp(-energy-1)")   #"-energy","-energy+1","exp(-energy-1)","exp(-energy)"
    return parser.parse_args()


def main(args):
    print(date)
    energyPolicy_training_data.append("Energy poilcy training")
    energyPolicy_training_data.append("Date:                                                          " + str(date))
    energyPolicy_training_data.append("Noise type:                                                    " + str(args.noise_type))
    energyPolicy_training_data.append("Policy Training max episodes:                                  " + str(args.iteration))
    energyPolicy_training_data.append("Number of iterations the energy model have ben trained:        " + str(args.model))
    energyPolicy_training_data.append("PPO gamma:                                                     " + str(args.gamma))
    energyPolicy_training_data.append("Do we add noise to sapair for calculating energy               " + str(args.sanoise))
    energyPolicy_training_data.append("The noise we add to sapair                                     " + str(args.noise_sigma))
    energyPolicy_training_data.append("h(energy)                                                      " + str(args.reward_function))
    energyPolicy_training_data.append(" \n\n")

    env = gym.make('CartPole-v0')
    Energy = Energy_net('energy', 'CartPole-v0')
    energy_saver = tf.train.Saver()

    sapairs = np.genfromtxt('training_data/sapairs.csv')
    noise_sapairs = np.genfromtxt('training_data/noise_sapairs.csv')

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter(args.logdir+'/'+args.alg, sess.graph)
        sess.run(tf.global_variables_initializer())
        if args.model == '':
            energy_saver.restore(sess, args.modeldir + '/' + args.alg + '/' + args.noise_type + '/' + 'model.ckpt')
        else:
            energy_saver.restore(sess,
                                 args.modeldir + '/' + args.alg + '/' + args.noise_type + '/' + 'model.ckpt-' + args.model)
        print("As for model after ", args.model, "training iterations")
        print("Energy for expert sapairs looks like:", Energy.get_energy(sapairs))
        print("Energy for noise sapairs (not corresponding to the noise trained for Energy) looks like:",
              Energy.get_energy(noise_sapairs))







        energyPolicy_training_data.append(["As for model after ", args.model, "training iterations"])

        energyPolicy_training_data.append("Energy for expert sapairs looks like:\n" + str(Energy.get_energy(sapairs)))
        energyPolicy_training_data.append(
            "Energy for noise sapairs (not corresponding to the noise trained for Energy) looks like:\n" + str(
                Energy.get_energy(noise_sapairs)))
        energyPolicy_training_data.append(" \n\n\n\n\n\n\n\n\n")
        energyPolicy_training_data.append("Done with reloading Energy. Start RL")

        # writer.close()

        open_file_and_save(args.logdir + '/' + args.model + "_iter_" + args.noise_type + '_Policy' + date,
                           energyPolicy_training_data)
        print("Done with reloading Energy. Start RL")








        # Start RL

        env.seed(0)
        ob_space = env.observation_space
        Policy = Policy_net('policy', env)
        Old_Policy = Policy_net('old_policy', env)
        PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
        saver = tf.train.Saver()

        # writer = tf.summary.FileWriter(args.logdir+'/'+args.noise_type, sess.graph)
        sess.run(tf.global_variables_initializer())
        obs = env.reset()

        reward = 0
        alter_reward = 0
        success_num = 0
        render = False
        #ep_reward = []

        # 用于记录每个trajectory的数据最后做总结
        Summary_after_max_episodes_training=[]
        Trajectory_rewards = []
        Trajectory_alter_rewards = []
        Trajectory_success_num = 0 # 与success_num一样，只不过这个不会清零，这个用于评估这个energy对于训练的效果

        plot_rewards = []
        plot_alter_rewards= []
        plot_iteration = []
        for iteration in range(args.iteration):
            observations = []
            actions = []
            v_preds = []
            rewards = []
            alter_rewards = []
            episode_length = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                episode_length += 1

                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                alter_rewards.append(alter_reward)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)

                # alter reward
                sapair = np.append(obs, np.array([[act]]), axis=1)
                # print("sapair:",sapair)
                energy = Energy.get_energy(sapair)[0][0]
                print("Energy for this sapair", energy)
                if args.sanoise == True:
                    # 定义 gauss noise 的均值和方差
                    mu, sigma = 0, args.noise_sigma
                    # 一维guass
                    # saNumber = sapairs.shape[0]
                    saShape = sapair.shape[1]
                    # sampleNo = saNumber * saShape  # 采样sampleNo个gauss noise
                    noise = np.random.normal(mu, sigma, saShape)
                    noise_sapair = sapair + noise
                    print("noise_sapair:",noise_sapair)
                    # noise_sapairs = np.reshape(noise_sapairs, newshape=[saNumber, saShape])
                    noise_energy = Energy.get_energy(noise_sapair)[0][0]
                    print("Noise Energy for this sapair", noise_energy)
                    energy = noise_energy

                if args.reward_function == "-energy":
                    alter_reward = -energy
                elif args.reward_function == "-energy+1":
                    alter_reward = -energy+1
                elif args.reward_function == "exp(-energy-1)":
                    alter_reward = np.exp(-energy-1)
                elif args.reward_function == "exp(-energy)":
                    alter_reward = np.exp(-energy)
                else:
                    print("No such reward_function")
                #alter_reward = np.exp(-energy-1)
                #alter_reward = -energy+1
                #alter_reward = reward
                #alter_reward = -energy


                # if render:
                # env.render()
                # pass
                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    alter_reward = -1

                    break
                else:
                    obs = next_obs

            # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)]), iteration)
            # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))]), iteration)






            # if sum(rewards) >= 195:
            #     success_num += 1
            #     Trajectory_success_num +=1
            #     render = True
            #     if success_num >= 100:
            #         saver.save(sess, args.savedir + '/model.ckpt')
            #         print('Clear!! Model saved.')
            #         break
            # else:
            #     success_num = 0

            sum_rewards = sum(rewards)
            sum_alter_rewards = sum(alter_rewards)
            Trajectory_rewards.append(sum_rewards)
            Trajectory_alter_rewards.append(sum_alter_rewards)
            #画图
            plot_rewards.append(sum_rewards)
            plot_alter_rewards.append(sum_alter_rewards)
            plot_iteration.append(iteration)
            #ep_reward.append(sum(rewards))

            # print("Sample done in one traj.")
            energyPolicy_training_data_for_this_episode = []
            energyPolicy_training_data_for_this_episode.append(" ")
            energyPolicy_training_data_for_this_episode.append("Trajectory:     " + str(iteration))
            energyPolicy_training_data_for_this_episode.append("episode_len:    " + str(episode_length))
            energyPolicy_training_data_for_this_episode.append("True rewards:   " + str(sum_rewards))
            energyPolicy_training_data_for_this_episode.append("alter_rewards:  " + str(sum_alter_rewards))
            open_file_and_save(args.logdir + '/' + args.model + "_iter_" + args.noise_type + '_Policy' + date,
                               energyPolicy_training_data_for_this_episode)
            print()
            print("Trajectory", iteration, ":")
            print("episode_len: ", episode_length)
            print("rewards: ", sum(rewards))
            print("alter_rewards: ", sum(alter_rewards))

            # gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = PPO.get_gaes(rewards=alter_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()
            rewards = np.array(rewards).astype(dtype=np.float32)
            alter_rewards = np.array(alter_rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, alter_rewards, v_preds_next]
            # inp = [observations, actions, gaes, rewards, v_preds_next]

            # train
            for epoch in range(6):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            # writer.add_summary(summary, iteration)
        # writer.close()

        #开始画图
        plt.title('Noise:'+str(args.sanoise))
        plt.plot(plot_iteration, plot_rewards, color='red', label='True_rewards')
        plt.plot(plot_iteration, plot_alter_rewards,color='green', label='alter_rewards')
        plt.legend() #显示图例

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.show()

        # Summary_after_max_episodes_training.append("After"+args.iterations+"episodes training")
        # Summary_after_max_episodes_training.append(" ")
        # Summary_after_max_episodes_training.append("Total Success numbers: " + str(Trajectory_success_num))
        # Summary_after_max_episodes_training.append("Mean True rewards of one episode:" + str(np.mean(Trajectory_rewards)))
        # Summary_after_max_episodes_training.append("Mean alter rewards of one episode:" + str(np.mean(Trajectory_alter_rewards)))
        #
        # open_file_and_save(args.logdir + '/' + args.model + "_iter_" + args.noise_type + '_Policy' + date,
        #                    energyPolicy_training_data_for_this_episode)



if __name__ == '__main__':
    args = argparser()
    main(args)
