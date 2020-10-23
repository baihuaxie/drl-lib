"""
Runner class
"""

from baselines.common.runners import AbstractRunner
from baselines.common.misc_util import dtype_to_torch

import torch

class Runner(AbstractRunner):
    """
    Object to run episodes in environment & returns statistics
    """
    def __init__(self, env, model, nsteps=3, gamma=1.0, lamb=1.0):
        """
        Constructor
        """
        super().__init__(env=env, model=model, nsteps=nsteps)
        # discount factor
        self.gamma = gamma
        # lambda for GAE
        self.lamb = lamb

    
    def _make_tensor(self, obs, reward, dones):
        """
        Convert environment returns to tensors

        Args:
            obs:        (int or list) observation
                        int for Discrete env, list of floats for Box env
            reward:     (float) reward
            dones:      (bool or list of bool) flag for if the env has terminated
                        bool for single env, list of bools for vectorized envs
        """
        if type(obs) == int:
            obs = [obs]
        _obs = torch.tensor(obs, dtype=dtype_to_torch(self.env.observation_space.dtype))
        _dones = torch.BoolTensor([dones])
        _reward = torch.tensor([reward], dtype=torch.float32)
        return _obs, _reward, _dones


    def run(self):
        """
        Run episode
        """
        # init mini-batch statistics
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogps = [],[],[],[],[],[]
        epinfos = []

        for _ in range(self.nsteps):
            # 1) step agent
            # runner superclass initialized self.obs = env.reset()
            # returned values are tensors
            action, value, neglogp = self.model.step(self.obs)
            mb_actions.append(action.detach().clone())
            mb_values.append(value.detach().clone())
            mb_neglogps.append(neglogp.detach().clone())

            # 2) step environment
            # Q1: how to handle done == True?
            # Q2: call env.reset() within runner or in main script?
            obs, reward, dones, info = self.env.step(action.item())
            self.obs, reward, self.dones = self._make_tensor(obs, reward, dones)
            mb_obs.append(self.obs.detach().clone())
            mb_rewards.append(reward)
            mb_dones.append(self.dones)
            if dones:
                break
        
        return map(tensor_stack, (mb_obs, mb_rewards, mb_actions, mb_values, \
            mb_dones, mb_neglogps))


def tensor_stack(lst):
    """
    stack a list of tensors into a tensor along dim=0 (timestep)
    """
    return torch.stack(lst, dim=0)


if __name__ == '__main__':
    import gym
    from baselines.common.policy_util import build_policy
    env = gym.make('CartPole-v0')
    policy = build_policy(env, 'mlp')
    runner = Runner(env=env, model=policy)
    env.reset()
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, \
        mb_neglogps = runner.run()    



        