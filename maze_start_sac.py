#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
from shutil import copy
import gym
import torch
import numpy as np
import random
import tensorflow as tf
from swoks import swoks as SWOKS
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from mbcd.sac_mbcd import SAC
from mbcd.envs.non_stationary_wrapper import NonStationaryEnv
from stable_baselines.common.callbacks import BaseCallback

def sac_swoks_maze(name, lr=0.0007, seed=123):
    config = Config()
    config.iteration_log_interval = 1
    weight_init_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.compat.v2.random.set_seed(seed)
    config.seed = weight_init_seed
    config.log_modulation = False
    config.cl_requires_task_label = True
    config.eval_task_fn = None
    config.cl_preservation = "scp"
    config.cl_loss_coeff = 0.5
    config.cl_n_slices = 200
    config.cl_alpha = 0.25
    config.expType = "tensorboard/"+name
    config.expID = "mdp_"+name+"_manlabels"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.deterministic_start_point = -1

    config.history_length = 1
    config.num_workers = 16
#    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, single_process=True)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr)
    config.lr = lr
    config.num_mini_batches = 5
    config.gradient_clip = 5
    config.ppo_ratio_clip = 0.1
    config.cl_num_tasks = 8
    swoks_conf = "swoks_conf_hc.json"
    config.max_steps = 750
    config.state_normalizer = None
    config.env = NonStationaryEnv(gym.envs.make("HalfCheetah-v2"), change_freq=40000)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL_MultiHead(
        int(np.prod(state_dim)), action_dim, config.cl_num_tasks, label_dim,
        phi_body=NatureConvBodySS(num_tasks=config.cl_num_tasks, discrete=False),
        actor_body=DummyBody_CL(16),
        critic_body=DummyBody_CL(16))

    config.reward_normalizer = RescaleNormalizer()
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.97
    config.entropy_weight = 0.01
    config.rollout_length = 32
    config.gradient_clip = 0.5
    config.logger = get_logger(log_dir=config.log_dir)
    config.rollout_schedule = [20000,50000,1,1]
    #copy maze json file for future references
    #copy(maze_conf_file_directory,config.log_dir)

    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.sac.policies import FeedForwardPolicy

    class CustomSACPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomSACPolicy, self).__init__(*args, **kwargs, layers=[256, 256], feature_extraction="mlp")

    model = SAC(CustomSACPolicy,
        env=config.env,
        rollout_schedule=config.rollout_schedule,
        verbose=0,
        batch_size=256,
        gradient_steps=20,
        target_entropy="auto",
        ent_coef="auto",
        swoks=SWOKS(swoks_conf, adopt=True, moreconf=config),
        max_std=0.5,
        num_stds=2.0,
        n_hidden_units_dynamics=200,
        n_layers_dynamics=4,
        dynamics_memory_size=100000,
        run_id=get_default_log_dir("hc"+str(seed)),
        tensorboard_log=config.log_dir,
        seed=0,
        weightfolder=config.log_dir,
        mbpo=False,
        mbcd=False
    )


    class TbCallback(BaseCallback):
        def __init__(self, verbose=0):
            self.is_tb_set = False
            super(TbCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            if not self.is_tb_set:
                with self.model.graph.as_default():
                    tf.summary.scalar('')

    model.learn(total_timesteps=480000, tb_log_name="sac-stuff")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=1)
    args = parser.parse_args()
    seed = int(args.seed)
    lr = 0.007
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(0)
    sac_swoks_maze('HalfCheetah-v2', lr=lr, seed=seed)
