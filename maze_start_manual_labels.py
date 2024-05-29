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
import gymnasium as gym
from swoks import swoks as SWOKS
import random
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def ppo_swoks_maze(name, lr=0.0007, user_env="ctgr", disc="cont", adopt="rand", confswoks=None, seed=123, oracle=False):
    config = Config()
    config.iteration_log_interval = 1
    weight_init_seed = seed
    random.seed(weight_init_seed)
    np.random.seed(weight_init_seed)
    torch.manual_seed(weight_init_seed)
    config.seed = weight_init_seed
    config.log_modulation = False
    maze_conf_file_directory="./mdp_graph.json"
    config.cl_requires_task_label = True
    config.eval_task_fn = None
    config.cl_preservation = "scp"
    config.cl_loss_coeff = 0.5
    config.cl_n_slices = 200
    config.cl_alpha = 0.25
    config.expType = "tensorboard/"+user_env
    # if disc=="cont" or disc is None:
    #     config.discrete = False
    # elif disc=="disc":
    #     config.discrete = True
    # else:
    #     ValueError("Wrong value for discretion: should be 'disc' or 'cont', but you gave "+disc)

    config.discrete=False
    config.adopt=False

    # if adopt=="rand" or None:
    #     config.adopt = False
    # elif adopt=="adopt":
    #     config.adopt = True
    # else:
    #     ValueError("Wrong value for adoption: should be 'rand' or 'adopt', but you gave "+adopt)
    config.expID = "mdp_"+user_env+confswoks+"-seedruns"
    config.log_dir = get_default_log_dir(config.expType) + config.expID + str(config.seed)
    config.deterministic_start_point = -1

    config.history_length = 1
    config.num_workers = 4
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, single_process=True)

    config.lr = lr
    config.num_mini_batches = 64
    #config.gradient_clip = 5
    config.ppo_ratio_clip = 0.1
    config.cl_num_tasks = 8
    config.entropy_weight = 0.01

    if user_env == "ctgr":
        if confswoks is None:
            swoks_conf = "swoks_conf.json"
        else:
            swoks_conf = confswoks
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr)
        config.max_steps = 750
        config.state_normalizer = ImageNormalizer()
        task_fn = lambda log_dir: CTgraph(name, maze_conf_file_directory, log_dir=config.log_dir)
        config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL_MultiHead(
            int(np.prod(state_dim)), action_dim, 2*config.cl_num_tasks, label_dim,
            phi_body=NatureConvBodySS(num_tasks=2*config.cl_num_tasks, discrete=config.discrete),
            actor_body=DummyBody_CL(16),
            critic_body=DummyBody_CL(16))

    if user_env == "mngrd":
        swoks_conf = "swoks_minigrid_conf.json"
        config.max_steps = 400
        lr=0.007
        config.lr= 0.007
        config.entropy_weight=0.1
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr)
        config.state_normalizer = ImageNormalizer(coef=10)
        task_fn = lambda log_dir: MiniGrid(log_dir=config.log_dir)
        config.lr=0.01
        config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL_MultiHead(
            state_dim, action_dim, config.cl_num_tasks, label_dim,
            phi_body=NatureConvBodySSmngrd(num_tasks=config.cl_num_tasks, in_channels=3),
            actor_body=DummyBody_CL(200),
            critic_body=DummyBody_CL(200))

    config.policy_fn = SamplePolicy
    config.reward_normalizer = RescaleNormalizer()
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.97

    config.rollout_length = 128
    config.gradient_clip = 5
    config.logger = get_logger(log_dir=config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)

    run_iterations_plus_swoks(WOKSPPOAgent(config), SWOKS(swoks_conf, adopt=config.adopt, moreconf=config), man_labels=oracle)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("-e", "--env", choices=["ctgr", "mngrd", "mnhk"], default="ctgr")
    parser.add_argument("-c", "--conf", help="path to SWOKS configuration file", default="swoks_conf.json")
    parser.add_argument("-lr", "--lr", default=0.007, type=float)
    parser.add_argument("-o", "--oracle", default=False, type=bool)

    args=parser.parse_args()
    lr = args.lr
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(0)
    user_env = args.env
    seed = int(args.seed)
    confpath = args.conf
    if user_env == "mnhk":
        ppo_swoks_maze('placeholder', lr=lr, user_env=user_env, confswoks=confpath, oracle=args.oracle)
    elif user_env == "ctgr":
        ppo_swoks_maze('CTgraph-v1', lr=lr, user_env=user_env, confswoks=confpath, seed=seed, oracle=args.oracle)
    elif user_env == "mngrd":
        ppo_swoks_maze('placeholder', lr=lr, user_env=user_env, confswoks="swoks_minigrid_conf.json", seed=seed, oracle=args.oracle)
    else:
        print("Bad env: "+user_env)
