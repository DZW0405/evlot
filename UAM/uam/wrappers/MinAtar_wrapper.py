import gym
import numpy as np


class MinAtar_wrapper(gym.ObservationWrapper):
    """将观察值的形状从（高度、宽度、输入通道数）改为（输入通道数、高度、宽度）"""
    def __init__(self, env):
        env.observation_space._shape = (env.observation_space.shape[2], *env.observation_space.shape[0:2])
        super().__init__(env)

    def observation(self, obs):        
        obs = np.moveaxis(obs, -1, 0)
        return obs
