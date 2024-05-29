#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import copy
import multiprocessing as mp
import sys
from .bench import Monitor
from ..utils import *
import uuid

class BaseTask:
    def __init__(self):
        pass

    def set_monitor(self, env, log_dir):
        if log_dir is None:
            return env
        mkdir(log_dir)
        return Monitor(env, '%s/%s' % (log_dir, uuid.uuid4()))

    def reset(self):
        #print 'base task reset called'
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        if done:
            next_state = self.env.reset()
        return next_state, reward, done, info

    def seed(self, random_seed):
        return self.env.seed(random_seed)

class MiniGrid(BaseTask):
    def __init__(self, log_dir=None):
        BaseTask.__init__(self)
        import gymnasium as gym
        from minigrid.wrappers import ReseedWrapper, ImgObsWrapper
        env1 = gym.make('MiniGrid-SimpleCrossingS9N3-v0')
        # env1 = RGBImgPartialObsWrapper(env1)
        env1 = ImgObsWrapper(env1)
        env2 = gym.make('MiniGrid-SimpleCrossingS9N2-v0')
        # env2 = RGBImgPartialObsWrapper(env2)
        env2 = ImgObsWrapper(env2)
        env3 = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
        # env3 = RGBImgPartialObsWrapper(env3)
        env3 = ImgObsWrapper(env3)
        envs_list = [env1, env2, env3]
        envs_list = [ReseedWrapper(env, (seed,)) for env, seed in zip(envs_list, [111,129,112])]
        self.envs_list = [self.set_monitor(env, log_dir) for env in envs_list]
        self.current_task = 0
        # self.env = self.envs_list[self.current_task]
        self.name = 'MiniGrid-SimpleCrossingS9N1-v0'
        # state = np.swapaxes(self.envs_list[self.current_task].reset()[0], 0,2)
        # self.env = self.envs_list[0]
        # self.env = self.set_monitor(env, log_dir)
        self.log_dir = log_dir
        self.observation_space = self.envs_list[0].observation_space
        self.action_dim = 3 #self.envs_list[0].action_space.n
        self.action_space = self.envs_list[0].action_space
        self.spec = self.envs_list[0].spec
        self.state_dim = self.envs_list[0].observation_space.shape
    
    def get_all_tasks(self, *args):
        return [{'task_label': np.array([0])},
                {'task_label': np.array([1])},
                {'task_label': np.array([2])}]

    def set_task(self, task):
        self.current_task = task['task_label'][0] % 3
        # self.current_task = (self.current_task + 1) % 3
        # self.env = self.envs_list[self.current_task]

    def reset_task(self, task=None):
        if task is not None:
            self.set_task(task)
        return self.reset()

    def reset(self):
        return np.swapaxes(self.envs_list[self.current_task].reset()[0], 0,2)
        #return np.swapaxes(self.env.reset()[0], 0,2)

    def step(self, action):
        state, reward, done, *info = self.envs_list[self.current_task].step(action)
        if done: state = self.reset()
        else: state = np.swapaxes(np.array(state), 0,2)
        return [np.array(state), reward, done, *info]


class CTgraph(BaseTask):
    def __init__(self, name, env_config_path, log_dir=None, flat=False):
        BaseTask.__init__(self)
        self.name = name
        import gymnasium as gym
        from gym_CTgraph import CTgraph_env
        from gym_CTgraph.CTgraph_conf import CTgraph_conf
        from gym_CTgraph.CTgraph_images import CTgraph_images
        env_config = CTgraph_conf(env_config_path)
        env_config = env_config.getParameters()
        imageDataset = CTgraph_images(env_config)
        env = gym.make(name, conf_data=env_config,images=imageDataset)
        self.env_config=env_config

        # state = env.init(env_config, imageDataset)[0]
        state = env.reset()
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(state))
        if flat: env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(np.shape(state)))
        self.observation_space = env.observation_space
        self.action_dim = env.action_space.n
        self.action_space = gym.spaces.Box(low=0, high=self.action_dim, shape=(1,), dtype=np.int)
        self.spec = env.spec
        if env_config['image_dataset']['1D']:
            self.state_dim = int(np.prod(env.observation_space.shape))
        else:
            self.state_dim = env.observation_space.shape

        self.env = self.set_monitor(env, log_dir)

        # get all tasks in graph environment instance
        from itertools import product
        depth = env_config['graph_shape']['d']
        branch = env_config['graph_shape']['b']
        tasks = list(product(list(range(branch)), repeat=depth))
        self.tasks = [{'goal': np.array(task)} for task in tasks]
        self.current_task = self.tasks[0]

    def step(self, action, flat=False):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        if self.env_config['image_dataset']['1D']: state = state.ravel()
        if flat:
            state = state/256
            state += np.full(state.shape, 1e-8)
        return np.array(state), reward, done, 0, info

    def reset(self, flat=False):
        state, _ = self.env.reset() # ctgraph returns state, reward, done, info in reset
        if self.env_config['image_dataset']['1D']: state = state.ravel()
        # return only state when reset is called to conform with other env in the repo
        if flat:
            state = state/256
            state += np.full(state.shape, 1e-8)
        return np.array(state)

    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()

    def set_task(self, taskinfo):
        self.env.unwrapped.set_high_reward_path(taskinfo['goal'])
        self.current_task = taskinfo

    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=False):
        if requires_task_label:
            tasks_label = np.eye(len(self.tasks)).astype(np.float32)
            tasks = copy.deepcopy(self.tasks)
            for task, label in zip(tasks, tasks_label):
                task['task_label'] = label
            return tasks
        else:
            return self.tasks

    def X(self):
        return self.env.X()

    def random_tasks(self, num_tasks, requires_task_label=True):
        tasks_idx = np.random.randint(low=0, high=len(self.tasks), size=(num_tasks,))
        if requires_task_label:
            all_tasks = copy.deepcopy(self.tasks)
            tasks_label = np.eye(len(all_tasks)).astype(np.float32)
            tasks = []
            for idx in tasks_idx:
                task = all_tasks[idx]
                task['task_label'] = tasks_label[idx]
                tasks.append(task)
            return tasks
        else:
            tasks = [self.tasks[idx] for idx in tasks_idx]

class ProcessTask:
    def __init__(self, task_fn, log_dir=None):
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = ProcessWrapper(worker_pipe, task_fn, log_dir)
        self.worker.start()
        self.pipe.send([ProcessWrapper.SPECS, None])
        self.state_dim, self.action_dim, self.name = self.pipe.recv()

    def step(self, action):
        self.pipe.send([ProcessWrapper.STEP, action])
        return self.pipe.recv()

    def reset(self):
        self.pipe.send([ProcessWrapper.RESET, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([ProcessWrapper.EXIT, None])

    def X(self):
        self.pipe.send([ProcessWrapper.CTGRAPH_X, None])
        return self.pipe.recv()

    def reset_task(self, task_info):
        self.pipe.send([ProcessWrapper.RESET_TASK, task_info])
        return self.pipe.recv()

    def set_task(self, task_info):
        self.pipe.send([ProcessWrapper.SET_TASK, task_info])

    def get_task(self):
        self.pipe.send([ProcessWrapper.GET_TASK, None])
        return self.pipe.recv()

    def get_all_tasks(self, requires_task_label):
        self.pipe.send([ProcessWrapper.GET_ALL_TASKS, requires_task_label])
        return self.pipe.recv()

    def random_tasks(self, num_tasks):
        self.pipe.send([ProcessWrapper.RANDOM_TASKS, num_tasks])
        return self.pipe.recv()

class ProcessWrapper(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    RESET_TASK = 4
    SET_TASK = 5
    GET_TASK = 6
    GET_ALL_TASKS = 7
    RANDOM_TASKS = 8
    CTGRAPH_X = 99
    def __init__(self, pipe, task_fn, log_dir):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.task_fn = task_fn
        self.log_dir = log_dir

    def run(self):
        np.random.seed()
        seed = np.random.randint(0, sys.maxsize)
        task = self.task_fn(log_dir=self.log_dir)
        task.seed(seed)
        while True:
            op, data = self.pipe.recv()
            if op == self.STEP:
                self.pipe.send(task.step(data))
            elif op == self.RESET:
                self.pipe.send(task.reset())
            elif op == self.EXIT:
                self.pipe.close()
                return
            elif op == self.SPECS:
                self.pipe.send([task.state_dim, task.action_dim, task.name])
            elif op == self.RESET_TASK:
                self.pipe.send(task.reset_task(data))
            elif op == self.SET_TASK:
                self.pipe.send(task.set_task(data))
            elif op == self.GET_TASK:
                self.pipe.send(task.get_task())
            elif op == self.GET_ALL_TASKS:
                self.pipe.send(task.get_all_tasks(data))
            elif op == self.RANDOM_TASKS:
                self.pipe.send(task.random_tasks())
            elif op == self.CTGRAPH_X:
                self.pipe.send(task.X())
            else:
                raise Exception('Unknown command')

class ParallelizedTask:
    def __init__(self, task_fn, num_workers, log_dir=None, single_process=False):

        if single_process:
            self.tasks = [task_fn(log_dir=log_dir) for _ in range(num_workers)]
        else:
            self.tasks = [ProcessTask(task_fn, log_dir) for _ in range(num_workers)]
        self.state_dim = self.tasks[0].state_dim
        self.action_dim = self.tasks[0].action_dim
        self.name = self.tasks[0].name
        self.single_process = single_process

    def step(self, actions):
        results = [task.step(action) for task, action in zip(self.tasks, actions)]
        try:
            results = map(lambda x: np.stack(x), zip(*results))
        except:
            import pdb; pdb.set_trace()
        return results

    def reset(self):
        results = [task.reset() for task in self.tasks]
        return np.stack(results)

    def close(self):
        if self.single_process:
            return
        for task in self.tasks: task.close()

    def X(self):
        results = [task.X() for task in self.tasks]
        return np.stack(results)

    def reset_task(self, task_info):
        results = [task.reset_task(task_info) for task in self.tasks]
        return np.stack(results)

    def set_task(self, task_info):
        for task in self.tasks:
            task.set_task(task_info)

    def get_task(self):
        return self.tasks[0].get_task()

    def get_all_tasks(self, requires_task_label):
        return self.tasks[0].get_all_tasks(requires_task_label)
    
    def random_tasks(self, num_tasks):
        return self.tasks[0].random_tasks(num_tasks)
