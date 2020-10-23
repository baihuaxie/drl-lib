"""
    subprocess vectorized environment class
"""

import multiprocessing as mp
import os
import torch
import numpy as np
from baselines.common.vec_env.vec_env import VecEnv
from baselines.common.vec_env.utils import obs_space_info, obs_dict_to_tensor
from baselines.common.misc_util import numpy_to_torch, dtype_to_torch


def worker_fn(server, customer, envs):
    """
    Worker function uses server.recv() to get commands & arguments, execute, then
    use server.send() to deliver the results through a Pipe object
    """
    if not isinstance(envs,(list, tuple)):
        envs = [envs]
    try:
        while True:
            cmd, data = server.recv()
            if cmd == 'get pid':
                server.send(os.getpid())
            elif cmd == 'reset':
                server.send([env.reset() for env in envs])
            elif cmd == 'step':
                # """ sub-routine to step environment """
                def step_env(env, action):
                    obs, reward, done, info = env.step(action)
                    if done:
                        obs = env.reset()
                    return obs, reward, done, info
                server.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'close':
                server.close()
                # break out while loop after current sub-process finishes
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('Interrupt!')
    finally:
        # close env for current sub-process
        [env.close() for env in envs]


class SubprocVecEnv(VecEnv):
    """
    A vectorized environment that runs in parallel in subprocesses
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Constructor

        Args:
            env_fns: (iterable of callable functions) each fn() in env_fns return a copy of
                     environment object
            context: (str) supported by multiprocessing; 'spawn', 'fork' or 'forkserver'
                     windows only supports 'spawn'
            in_series: (int) number of environment copies to be run in series by a single
                       sub-process
        """
        envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)

        ctx = mp.get_context(context)
        assert self.num_envs % in_series == 0, "Number of environment copies must be \
            divisible by number of environments to run in series"
        self.num_servers = self.num_envs // in_series

        # split envs into chunks of equal size to be allocated to each sub-process
        self.envs = [envs[i:i+in_series] for i in range(0, self.num_envs, in_series)]

        # create a Pipe between a list of servers and a list of customers
        self.customers, self.servers = zip(*[ctx.Pipe() for _ in range(self.num_servers)])
        # subprocess
        # Q: somehow need to pass customer to subprocess (otherwise error), why?
        self.ps = [ctx.Process(target=worker_fn, args=(server, customer, envs)) for \
            (server, customer, envs) in zip(self.servers, self.customers, self.envs)]
        for p in self.ps:
            # terminate subprocess if main process exits
            p.daemon = True
            p.start()

        # Q: calling customer.close() here?

        self.closed = False
        self.waiting = False

        env = envs[0]
        obs_space = env.observation_space
        # observation_space info dicts
        # Q: do this in master proc, or call sub-proc?
        self._obs_keys, self._obs_shapes, self._obs_dtypes = obs_space_info(obs_space)
        # store observations from each environment copy in a dict of
        # {key: tensor(num_envs x subspace_shape)}
        self._obs_dict = {k: torch.zeros((self.num_envs,) + tuple(self._obs_shapes[k]), \
            dtype=dtype_to_torch(self._obs_dtypes[k])) for k in self._obs_keys}

        self._actions = None
        self._rews = torch.zeros((self.num_envs,), dtype=torch.float32)
        self._dones = torch.BoolTensor([True for _ in range(self.num_envs)])
        self._infos = [{} for _ in range(self.num_envs)]

        super().__init__(self.num_envs, obs_space, env.action_space)


    def reset(self):
        """
        """
        assert not self.closed
        for customer in self.customers:
            customer.send(('reset', None))
        obs = [customer.recv() for customer in self.customers]
        obs = _flatten_list(obs)
        for e, ob in enumerate(obs):
            self._save_obs(e, ob)
        return obs_dict_to_tensor(self._obs_dict)

    
    def step_async(self, actions):
        """
        Obtain action(s) for sub-process environment copies asynchronously

        Args:
            actions: (torch.tensor)
        """
        assert not self.closed
        self._actions = np.array_split(actions.numpy(), self.num_servers)
        for customer, action in zip(self.customers, self._actions):
            customer.send(('step', action))
        self.waiting = True


    def step_wait(self):
        """
        """
        assert not self.closed
        results = [customer.recv() for customer in self.customers]
        self.waiting = False
        # results is a 2D list of structure [[(obs,rew,dones,info),...],...]
        # dim=0 -> self.num_servers
        # dim=1 -> # of environment copies run by each sub-process
        results = _flatten_list(results)
        for e, result in enumerate(results):
            obs, rew, done, self._infos[e] = result
            self._save_obs(e, obs)
            self._rews[e] = numpy_to_torch(rew).detach().clone()
            self._dones[e] = numpy_to_torch(done).detach().clone()
        return obs_dict_to_tensor(self._obs_dict), self._rews, self._dones, self._infos


    def close(self):
        """
        Close all sub-processes
        """
        self.closed = True
        if self.waiting:
            # remaining sub-proesses waiting to receive results
            for customer in self.customers:
                customer.recv()
        for customer in self.customers:
            # close sub-process
            customer.send(('close', None))
        # join() sub-processes
        [p.join() for p in self.ps]


    def get_images(self):
        """
        """

    def get_pid(self):
        """
        dummy method for test
        returns subprocess id
        """
        pids = []
        for customer in self.customers:
            customer.send(('get pid', None))
            pids.append(customer.recv())
        return pids

    def _save_obs(self, idx, obs):
        """
        Save observations from each environment copy

        Args:
            idx: (int) subenv index in num_envs
            obs: (tensor or dict of tensors) observations from subenv[idx]
                 - a single tensor for simple environments (e.g., Discrete)
                 - a dict of tensors indexed by keys of Dict / Tuple of subenv[idx]
        """
        obs = numpy_to_torch(obs).detach().clone()
        for key in self._obs_keys:
            if key is None:
                # simple envs
                self._obs_dict[key][idx] = obs
            else:
                self._obs_dict[key][idx] = obs[key]


def _flatten_list(lst):
    """
    flatten a list or tuple of structure [[(obs,rew,dones,info),...],...] into
    a list of [(obs,rew,dones,info),(...),...]
    
    so that can use zip(*lst) to unpack
    """
    assert isinstance(lst, (list, tuple))
    return [lst_2 for lst_1 in lst for lst_2 in lst_1]

if __name__ == '__main__':
    import gym
    env = gym.make('FrozenLake-v0')
    num_envs = 5
    def seed_env(env, seed):
        env.seed(seed)
        return lambda: env
    env_fns = [seed_env(env, idx) for idx in range(num_envs)]
    venv = SubprocVecEnv(env_fns)
    
    venv.reset()

    venv.get_pid()
