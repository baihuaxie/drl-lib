"""
    A dummy vectorized environment, runs copies of envs sequentially in a single
    process. Useful for debug purposes.
"""

import torch

from .vec_env import VecEnv
from .utils import obs_space_info, obs_dict_to_tensor
from baselines.common.misc_util import dtype_to_torch, numpy_to_torch

class DummyVecEnv(VecEnv):
    """
    Dummy vectorized environment class

    use:
    - debug
    - num_env = 1 (avoid communication overhead of multiprocessing)
    """
    def __init__(self, env_fns):
        """
        Constructor

        Args:
            env_fns: (iterable of callable function) an iterable of callale functions;
                     each function returns a copy of the environment object(s)
        
        Note:
        - self.envs contains copies of the same env object, accessible by self.env[i]
        - env_fn() always return a single env object, but may contain single or multiple
          observation_space objects, known as subspaces:
            - env.observation_space == simple gym.spaces objects like Discrete or Box
                - contains a single subspace
            - env.observation_space == iterable gym.spaces objects like Dict or Tuple
                - contains multiple subspaces accessible by env.observation_space.spaces
        - obs_space_info(env) returns a tuple of (keys, shapes, dtypes) taht allows:
            - each subspace to be indexed by a key
            - e.g., 'position': Discrete, 'velocity': Box, etc.
            - for single-subspace object like Discrete, use a null key (None)
        - each subspace copy in the vectorized env class is therefore accessible by
          its key + its index in num_envs
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        env = self.envs[0]
        obs_space = env.observation_space

        super().__init__(self.num_envs, obs_space, env.action_space)
    
        # observation_space info dicts
        self._obs_keys, self._obs_shapes, self._obs_dtypes = obs_space_info(obs_space)
        # store observations from each environment copy in a dict of
        # {key: tensor(num_envs x subspace_shape)}
        # e.g., env.observation_space = gym.Dict with 3 keys for 3 sub-spaces;
        # vectorized for 10 copies, then:
        # self._obs_dict is a dict of 3 key-tensor pairs, each tensor is 10 x subspace.shape
        self._obs_dict = {k: torch.zeros((self.num_envs,) + tuple(self._obs_shapes[k]), \
            dtype=dtype_to_torch(self._obs_dtypes[k])) for k in self._obs_keys}

        self._actions = None

        self._rews = torch.zeros((self.num_envs,), dtype=torch.float32)
        self._dones = torch.BoolTensor([True for _ in range(self.num_envs)])
        self._infos = [{} for _ in range(self.num_envs)]


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
 
    
    def reset(self):
        """
        """
        for idx in range(self.num_envs):
            obs = self.envs[idx].reset()
            self._save_obs(idx, obs)
        return obs_dict_to_tensor(self._obs_dict)


    def step_async(self, actions):
        """
        Steps environment copies asynchronously

        Args:
            actions: (list of tensors or np.ndarrays) each environment copy receives
                     its own action

        Note:
        naively, actions could be:
        1) a 0D (scalar) tensor -- self.num_envs==1 & action_space.shape is scalar
        2) a xD tensor -- self.num_envs==1 & action_space.shape is a xD tensor
        3) a list (len could be 1) of 0D/xD tensors -- self.num_envs>=1

        len(actions) == self.num_envs would fail as a correct condition if actions is
        a single tensor
        to simplify code, require that argument being passed to be a list of tensors;
        never pass a single xD tensor to step() function
        """
        if isinstance(actions, list):
            assert len(actions) == self.num_envs, "Expected actions list to have {} \
                elements but got {} elements".format(self.num_envs, len(actions))
            self._actions = actions
        else:
            assert self.num_envs == 1, "Expected actions {} to be list type but \
                got {} instead".format(actions, type(actions))
            self._actions = [actions]
        # conver to (if any) torch.tensor type action(s) to np.ndarray
        for idx, action in enumerate(self._actions):
            if isinstance(action, torch.Tensor):
                self._actions[idx] = action.numpy()
        return self._actions


    def step_wait(self):
        """
        """
        for e in range(self.num_envs):
            action = self._actions[e]
        
            obs, rew, done, self._infos[e] = self.envs[e].step(action)
            self._save_obs(e, obs)
            self._rews[e] = numpy_to_torch(rew).detach().clone()
            self._dones[e] = numpy_to_torch(done).detach().clone()
        return self._obs_dict, self._rews, self._dones, self._infos


    def get_images(self):
        """
        """
        return [env.render(mode='rgb_array') for env in self.envs]
        


