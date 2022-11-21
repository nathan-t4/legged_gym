from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class A1Tail(LeggedRobot):
    def _get_noise_scale_vec(self, cfg): # modify to change action size to 15. size should be 244\
        """ @override 
            (num_obs = 12, size(noise_vec) = 238) --> (num_obs = 15, size(noise_vec) = 244)
        """
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:27] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[27:42] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[42:57] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[57:244] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        print(f"noise scale vec: {noise_vec}")
        print(f"Size of noise_vec: {noise_vec.size()}")
        print(f"Size of self.obs_buf: {self.obs_buf.size()}")
        return noise_vec

    def _reward_powers(self):
        # Penalize powers = torques * joint velocities
        # print(self.torques * self.dof_vel)
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
    
