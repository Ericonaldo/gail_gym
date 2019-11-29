import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.behavior_clone import BehavioralCloning
from algo.ppo import PPOTrain
from tools import kl_divergence
import matplotlib 
import matplotlib.pyplot as plt

exp_len=200

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/bc_reward')
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/bc_reward')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e3), type=int)  # BC学习的此书
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    BCPolicy = Policy_net('bcpolicy', env)
    BC = BehavioralCloning(BCPolicy)
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep) #实例化一个Saver对象，在训练过程中，定期调用saver.save方法，像文件夹中写入包含当前模型中所有可训练变量的checkpoint文件 saver.save(sess,FLAGG.train_dir,global_step=step)

    exp_obs = np.genfromtxt('trajectory/observations.csv')[0:exp_len]   #exp_len=200
    exp_acts = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)[0:exp_len]

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph) #指定一个文件用来保存图。格式：tf.summary.FileWritter(path,sess.graph)，可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中
        sess.run(tf.global_variables_initializer())

        inp = [exp_obs, exp_acts]  #inp[0]就是observations， inp[1]就是actoins
        
        for iteration in range(args.iteration):  # episode

            # train
            for epoch in range(args.epoch_num):
                # select sample indices in [low, high)
                sample_indices = np.random.randint(low=0, high=exp_obs.shape[0], size=args.minibatch_size)   #函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)

                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                BC.train(obs=sampled_inp[0], actions=sampled_inp[1])

            bc_summary = BC.get_summary(obs=inp[0], actions=inp[1])

            if (iteration+1) % args.interval == 0:
                saver.save(sess, args.savedir + '/model.ckpt', global_step=iteration+1)

            writer.add_summary(bc_summary, iteration)


        print("Done with BC. Start RL")
        # Start RL
        obs = env.reset()
        ob_space = env.observation_space
       
        reward = 0
        alter_reward = 0
        success_num = 0
        render = False
        ep_reward=[]
        for iteration in range(5*args.iteration):
            print("iter:{}".format(iteration))
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
                alter_reward = np.log(1/(kl_divergence(obs, BCPolicy, Policy)+0.00001))
                #alter_reward = -kl_divergence(obs, BCPolicy, Policy)
                #alter_reward = kl_divergence(obs, BCPolicy, Policy)
                #print(alter_reward)
                if render:
                    #env.render()
                    pass
                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    alter_reward = -1
                    print("episode_len: ",episode_length)
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= 195:
                success_num += 1
                render = True
                if success_num >= 100:
                    saver.save(sess, args.savedir+'/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0
            ep_reward.append(sum(rewards))
            print("rewards: ",sum(rewards))
            print("alter_rewards: ",sum(alter_rewards))
            print("Sample done in one traj.")
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

            print("Begin Training")
            # train
            for epoch in range(6):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                PPO.train(obs=sampled_inp[0],
                        actions=sampled_inp[1],
                        gaes=sampled_inp[2],
                        rewards=sampled_inp[3],
                        v_preds_next=sampled_inp[4])
                """
                summary = PPO.get_summary(obs=inp[0],
                        actions=inp[1],
                        gaes=inp[2],
                        rewards=inp[3],
                        v_preds_next=inp[4])
                """

            #writer.add_summary(summary, iteration)
        writer.close()
    plt.plot(ep_reward)

if __name__ == '__main__':
    args = argparser()
    main(args)
