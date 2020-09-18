import gym
import atari_py

#env = gym.make('CartPole-v0')
env = gym.make('SpaceInvaders-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    
env.close()

print(atari_py.list_games())