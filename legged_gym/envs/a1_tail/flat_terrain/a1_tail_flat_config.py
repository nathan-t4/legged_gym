# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.a1_tail.rough_terrain.a1_tail_rough_config import A1TailRoughCfg, A1TailRoughCfgPPO

class A1TailFlatCfg( A1TailRoughCfg ):
    class env( A1TailRoughCfg.env ):
        num_observations = 57 # flat: 57, rough: 244

    class terrain( A1TailRoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False # False: blind robot

    class commands( A1TailRoughCfg.commands):
        class ranges (A1TailRoughCfg.commands.ranges):
            ang_vel_yaw = [0, 0]

  
    class rewards( A1TailRoughCfg.rewards ):
        class scales( A1TailRoughCfg.rewards.scales ):
            # Rewards in use.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5   
            powers = -0.0001

            # Set remaining reward terms to zero.
            termination = -0.0        
            lin_vel_z = 0.0
            ang_vel_xy = -0.0
            orientation = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_acc = -0.0
            base_height = -0. 
            feet_air_time =  0.0
            collision = -0.
            feet_stumble = -0.0 
            action_rate = -0.0
            stand_still = -0.
        
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.1 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.

class A1TailFlatCfgPPO( A1TailRoughCfgPPO ):
    class algorithm( A1TailRoughCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( A1TailRoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_a1_tail'

  