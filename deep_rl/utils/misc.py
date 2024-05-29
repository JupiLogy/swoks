#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import tensorboardX
import datetime
import torch
from copy import deepcopy
from .torch_utils import *
#from io import BytesIO
#import scipy.misc
#import torchvision

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

def run_iterations_plus_swoks(agent, swoks, man_labels=False):
    from ..shell_modules.mmn.ssmask_utils import set_model_task, set_mask, get_mask
    config = agent.config
    random_seed(config.seed)
    agent_name = agent.__class__.__name__


    # To print the structure of phi_body
    print("phi_body structure:")
    print(agent.network.network.phi_body)

    # To print the structure of actor_body
    print("\nactor_body structure:")
    print(agent.network.network.actor_body)

    # To print the structure of critic_body
    print("\ncritic_body structure:")
    print(agent.network.network.critic_body)

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    tasks_info = agent.task.get_all_tasks(config.cl_requires_task_label)
    labels_set = np.eye(8) #[task['task_label'] for task in tasks_info]
    task_label = None
    visited_tasks = [[]]
    backup = [None, None, 0]
    for doing_each_task_twice in [1, 2]:
        for task_idx, task_info in reversed(list(enumerate(tasks_info))):
            print('\nstart training on task {0}'.format(task_idx))
            print(task_info)
            # states = np.array([x[0] for x in agent.task.reset_task(task_info)])
            states = np.array([x for x in agent.task.reset_task(task_info)])
            if type(states[0,-1]) == str:
                states = [y[0] for y in states]
            if len(states[0].shape)<3:
                states = config.state_normalizer(np.expand_dims(states, axis=1))
            else:
                states = config.state_normalizer(states)
            agent.states = states
            while True:
                # Backing up
                if iteration%50 == 0 and task_label is not None:
                    if man_labels:
                        mask = deepcopy(get_mask(agent.network,
                                        np.where(np.array(task_label)==1)[0][0]))
                    else:
                        mask = deepcopy(get_mask(agent.network, swoks.current_task))
                    if iteration%100 == 0:
                        backup = [mask, backup[1], 1]
                        print("backup0")
                    elif iteration%100 == 50:
                        backup = [backup[0], mask, 0]
                        print("backup1")

                # Task change stuff
                change_task=False
                if man_labels:
                    if task_label is not None and\
                       task_label != task_info['task_label'].tolist():
                        change_task = True
                elif swoks.task_changing:
                    change_task = True
                    swoks.task_changing = False

                if change_task:
                    print("changing task.")
                    print(task_label)
                    print("loading" + str(backup[2]))
                    try:
                        set_mask(agent.network, deepcopy(backup[backup[2]]),
                                 np.where(np.array(task_label)==1)[0][0])
                    except:
                        print("premature task switching. Loading first mask.")
                        set_mask(agent.network, deepcopy(backup[0]),
                                 np.where(np.array(task_label)==1)[0][0])
                learn=True
                if not man_labels and swoks.tested_tasks!=[]:
                    learn=False
                if man_labels:
                    task_label = task_info['task_label'].tolist()
                    set_model_task(agent.network,
                                   np.where(task_info['task_label']==1)[0][0])
                else:
                    #import pdb; pdb.set_trace()
                    task_label = np.eye(len(labels_set))[swoks.current_task]
                    set_model_task(agent.network, swoks.current_task)

                if swoks.new_agent and swoks.adopt_masks:
                    swoks.new_agent = False
                    set_mask(agent.network, deepcopy(get_mask(agent.network, swoks.current_task-1)), swoks.current_task)
                    # adopt the previous guy's mask!
                    print(f"----\n ADOPTING MASK {swoks.current_task-1} FOR AGENT {swoks.current_task}\n----")

                # Iterating
                actions_list, states_info_list, reward_list, values_list = agent.iteration(\
                    task_label=task_label, labels_set=labels_set, learn=learn)

                # states = np.array([x[0] for x in agent.task.reset_task(task_info)])
                states = np.array([x for x in agent.task.reset_task(task_info)])
                if len(states[0].shape)<3:
                    states = config.state_normalizer(np.expand_dims(states, axis=1))
                else:
                    states = config.state_normalizer(states)
                agent.states = states

                actions_list = np.squeeze(actions_list)
                #SWOKS code here
                for i in range(len(actions_list)): #for each agent
                    for j in range(len(actions_list[0])): #for each timestep in rollout
                        swoks.step([values_list[0][j][i],values_list[1][j][i],values_list[2][j][i],\
                            0], # values_list[3][j][i]],\
                            reward_list[i][j], actions_list[i][j], supp=states_info_list[i][j])

                steps.append(agent.total_steps)
                rewards.append(np.mean(agent.last_episode_rewards))
                if iteration % config.iteration_log_interval == 0:
                    config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f, %f' % (
                        iteration, agent.total_steps,
                        np.mean(agent.last_episode_rewards),
                        np.max(agent.last_episode_rewards),
                        np.min(agent.last_episode_rewards),
                        agent.loss
                    ))
                    config.logger.scalar_summary('avg reward',
                                                 np.mean(agent.last_episode_rewards))
                    config.logger.scalar_summary('max reward',
                                                 np.max(agent.last_episode_rewards))
                    config.logger.scalar_summary('min reward',
                                                 np.min(agent.last_episode_rewards))
                    config.logger.scalar_summary('loss', agent.loss)

                    config.logger.multipoint_summary("Historical", {"hTask1":swoks.emd_val[0][-1], "hTask2":swoks.emd_val[1][-1],\
                                                                    "hTask3":swoks.emd_val[2][-1], "hTask4":swoks.emd_val[3][-1]})
                    config.logger.multipoint_summary("PVal", {"hTask1":swoks.pval[0], "hTask2":swoks.pval[1], "hTask3":swoks.pval[2], "hTask4":swoks.pval[3]})

                if iteration % (config.iteration_log_interval * 100) == 0:
                    with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                        (agent_name, config.tag, agent.task.name), 'wb') as f:
                        pickle.dump({'rewards': rewards, 'steps': steps}, f)
                    agent.save(config.log_dir +\
                               '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                                                        agent.task.name))
                    for tag, value in agent.network.named_parameters():
                        tag = tag.replace('.', '/')
                        config.logger.histo_summary(tag, value.data.cpu().numpy())
                iteration += 1
                if config.max_steps and iteration % config.max_steps == 0:
                    with open(config.log_dir +\
                              '/%s-%s-online-stats-%s-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, task_idx+1),
                              'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)
                    agent.save(config.log_dir +\
                               '/%s-%s-model-%s-task-%d.bin'%(agent_name,
                                                              config.tag,
                                                              agent.task.name,
                                                              task_idx+1))
                    task_start_idx = len(rewards)
                    break
    agent.close()
    return steps, rewards


def run_iterations_plus_mbcd(agent, man_labels=False): #run iterations continual learning (mulitple tasks) setting
    # from ..shell_modules.mmn.ssmask_utils import set_model_task, set_mask, get_mask
    config = agent.config
    random_seed(config.seed)
    agent_name = agent.__class__.__name__

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    tasks_info = agent.task.get_all_tasks(config.cl_requires_task_label)
    labels_set = [task['task_label'] for task in tasks_info]
    print(labels_set)
    task_label = None
    visited_tasks = [[]]
    backup = [None, None, 0]
    for doing_each_task_twice in [1, 2]:
        for task_idx, task_info in reversed(list(enumerate(tasks_info))):
            print('\nstart training on task {0}'.format(task_idx))
            print(task_info)
            states = np.array([x[0] for x in agent.task.reset_task(task_info)])
            if len(states[0].shape)<3:
                states = config.state_normalizer(np.expand_dims(states, axis=1))
            else:
                states = config.state_normalizer(states)
            agent.states = states
            while True:
                # # Backing up
                # if iteration%50 == 0 and task_label is not None:
                #     if man_labels:
                #         mask = deepcopy(get_mask(agent.network, np.where(np.array(task_label)==1)[0][0]))
                #     else:
                #         mask = deepcopy(get_mask(agent.network, swoks.current_task))
                #     if iteration%100 == 0:
                #         backup = [mask, backup[1], 1]
                #         print("backup0")
                #     elif iteration%100 == 50:
                #         backup = [backup[0], mask, 0]
                #         print("backup1")

                # # Task change stuff
                # change_task=False
                # if man_labels:
                #     if task_label is not None and task_label != task_info['task_label'].tolist():
                #         change_task = True
                # elif swoks.task_changing:
                #     change_task = True
                #     swoks.task_changing = False

                # if change_task:
                #     print("changing task.")
                #     print(task_label)
                #     print("loading" + str(backup[2]))
                #     set_mask(agent.network, deepcopy(backup[backup[2]]), np.where(np.array(task_label)==1)[0][0])
                # learn=True
                # if not man_labels and swoks.tested_tasks!=[]:
                #     learn=False
                # if man_labels:
                #     task_label = task_info['task_label'].tolist()
                #     set_model_task(agent.network, np.where(task_info['task_label']==1)[0][0])
                # else:
                #     task_label = np.eye(4)[swoks.current_task]
                #     set_model_task(agent.network, swoks.current_task)

                # Iterating
                actions_list, states_info_list, reward_list, values_list = agent.iteration(\
                    task_label=task_label, labels_set=labels_set)

                states = np.array([x[0] for x in agent.task.reset_task(task_info)])
                if len(states[0].shape)<3:
                    states = config.state_normalizer(np.expand_dims(states, axis=1))
                else:
                    states = config.state_normalizer(states)
                agent.states = states

                actions_list = np.squeeze(actions_list)
                # #SWOKS code here
                # for i in range(len(actions_list)): #for each agent
                #     for j in range(len(actions_list[0])): #for each timestep in rollout
                #         swoks.step([values_list[0][j][i],values_list[1][j][i],values_list[2][j][i],values_list[3][j][i]], reward_list[i][j], actions_list[i][j], supp=states_info_list[i][j])

                steps.append(agent.total_steps)
                rewards.append(np.mean(agent.last_episode_rewards))
                if iteration % config.iteration_log_interval == 0:
                    config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f' % (
                        iteration, agent.total_steps, np.mean(agent.last_episode_rewards),
                        np.max(agent.last_episode_rewards),
                        np.min(agent.last_episode_rewards)
                    ))
                    config.logger.scalar_summary('avg reward', np.mean(agent.last_episode_rewards))
                    config.logger.scalar_summary('max reward', np.max(agent.last_episode_rewards))
                    config.logger.scalar_summary('min reward', np.min(agent.last_episode_rewards))
                    config.logger.scalar_summary('model',      agent.deepMBCD.current_model)
                    config.logger.multipoint_summary('Statistic', {"s1":agent.deepMBCD.S[0], "snew":agent.deepMBCD.S[-1]})
                    # config.logger.multipoint_summary("Historical", {"hTask1":swoks.emd_val[0][-1], "hTask2":swoks.emd_val[1][-1],\
                    #                                                 "hTask3":swoks.emd_val[2][-1], "hTask4":swoks.emd_val[3][-1]})
                    # config.logger.multipoint_summary("PVal", {"Task1":swoks.pval[0], "hTask2":swoks.pval[1], "hTask3":swoks.pval[2], "hTask4":swoks.pval[3]})

                if iteration % (config.iteration_log_interval * 100) == 0:
                    with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                        (agent_name, config.tag, agent.task.name), 'wb') as f:
                        pickle.dump({'rewards': rewards, 'steps': steps}, f)
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    for tag, value in agent.network.named_parameters():
                        tag = tag.replace('.', '/')
                        config.logger.histo_summary(tag, value.data.cpu().numpy())
                iteration += 1
                if config.max_steps and iteration % config.max_steps == 0:
                    with open(config.log_dir + '/%s-%s-online-stats-%s-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, task_idx+1), 'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)
                    agent.save(config.log_dir + '/%s-%s-model-%s-task-%d.bin'%(agent_name, config.tag, \
                        agent.task.name, task_idx+1))
                    task_start_idx = len(rewards)
                    break
    agent.close()
    return steps, rewards

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return './log/%s/%s' % (name, get_time_str())

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
