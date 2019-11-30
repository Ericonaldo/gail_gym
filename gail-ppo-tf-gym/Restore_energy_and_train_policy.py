import gym
import numpy as np
import tensorflow as tf
import argparse
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain
from network_models.energy_net import Energy_net
from tools import kl_divergence
import matplotlib
import matplotlib.pyplot as plt
render = True

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='directory of model', default='trained_models')
    parser.add_argument('--alg', help='chose algorithm one of gail, ppo, bc, kl_bc, energy', default='energy')
    parser.add_argument('--noise_type', help='chose noise type for energy model(new_noise or fixed_noise)', default='fixed_noise')
    parser.add_argument('--model', help='number of model to test. model.ckpt-number', default='600')
    parser.add_argument('--logdir', help='log directory', default='log/test/energy_policy')
    parser.add_argument('--iteration', default=int(1e7))
    parser.add_argument('--stochastic', action='store_false')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    Energy = Energy_net('energy', 'CartPole-v0')
    energy_saver = tf.train.Saver()

    sapairs = np.genfromtxt('training_data/sapairs.csv')
    noise_sapairs = np.genfromtxt('training_data/noise_sapairs.csv')

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter(args.logdir+'/'+args.alg, sess.graph)
        sess.run(tf.global_variables_initializer())
        if args.model == '':
            energy_saver.restore(sess, args.modeldir+'/'+args.alg+'/'+args.noise_type+'/'+'model.ckpt')
        else:
            energy_saver.restore(sess, args.modeldir+'/'+args.alg+'/'+args.noise_type+'/'+'model.ckpt-'+args.model)
        print("As for model after ", args.model,"training iterations")
        print("Energy for expert sapairs looks like:",Energy.get_energy(sapairs))
        print("Energy for noise sapairs looks like:", Energy.get_energy(noise_sapairs))



        # writer.close()



        print("Done with reloading Energy. Start RL")
        # Start RL

        env.seed(0)
        ob_space = env.observation_space
        Policy = Policy_net('policy', env)
        Old_Policy = Policy_net('old_policy', env)
        PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
        saver = tf.train.Saver()



        writer = tf.summary.FileWriter(args.logdir+'/'+args.noise_type, sess.graph)
        sess.run(tf.global_variables_initializer())
        obs = env.reset()

        reward = 0
        alter_reward = 0
        success_num = 0
        render = False
        ep_reward=[]

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

                #alter reward
                sapair = np.append(obs,np.array([[act]]), axis=1)
                #print("sapair:",sapair)
                energy = Energy.get_energy(sapair)[0][0]
                #print("Energy for this sapair", energy)

                alter_reward = -energy


                #if render:
                #env.render()
                    # pass
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
                    saver.save(sess, args.savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            ep_reward.append(sum(rewards))
            # print("Sample done in one traj.")
            print("Trajectory", iteration,":")
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
            #inp = [observations, actions, gaes, rewards, v_preds_next]

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

            writer.add_summary(summary, iteration)
        writer.close()















if __name__ == '__main__':
    args = argparser()
    main(args)
