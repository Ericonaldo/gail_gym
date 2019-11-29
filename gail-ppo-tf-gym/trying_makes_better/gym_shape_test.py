import gym

env = gym.make('Acrobot-v1')
action_space = env.action_space
observation_space = env.observation_space

print("a_space=",action_space)
print(action_space.shape)

print("o_space=", observation_space)
print(observation_space.shape[0])