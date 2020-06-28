import gym
import sys

from gym import envs
envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)

# env = gym.make(sys.argv[1])
# env = gym.make('LunarLander-v2')
env = gym.make('Acrobot-v1')



env.reset()
for i in range(1000):
# for i in range(10):

    env.render()
    ob, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()