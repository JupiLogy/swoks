#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
import torch

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim +1)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim +1)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, to_numpy=False):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None):
        obs = tensor(obs)
        if len(obs.shape) > 4:
            obs = torch.squeeze(obs, dim=1)
        elif len(obs.shape) < 4:
            obs = obs[:, None, :, :]
        phi = self.network.phi_body(obs)[0]
        if task_label is not None:
            # if len(task_label.shape) > 2:
            #     task_label = task_label[0]
            phi_lab = torch.cat((phi, torch.tensor([task_label]).expand(len(phi), -1).to(torch.device("cuda"))), 1)
            phi_a = self.network.actor_body(phi_lab)
            phi_v = self.network.critic_body(phi_lab)
        else:
            phi_a = self.network.actor_body(phi)
            phi_v = self.network.critic_body(phi)
        try:
            logits = self.network.fc_action(phi_a.float())
        except:
            import pdb; pdb.set_trace()
        v = self.network.fc_critic(phi_v.float())
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, phi
    

class CategoricalActorCriticNetSS(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)
        from ..shell_modules.mmn.ssmask_utils import set_model_task
        self.set_model_task = set_model_task

    def predict(self, obs, action=None, task_label=None):
        obs = tensor(obs)
        if len(obs.shape) > 4:
            obs = torch.squeeze(obs, dim=1)
        elif len(obs.shape) < 4:
            obs = obs[:, None, :, :]

        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)

        task_id = self._task_label_to_id(task_label)

        self.set_model_task(self.network, task_id)

        phi = self.network.phi_body(obs)[0]
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        v = self.network.fc_critic(phi_v.float())
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, phi

class ActorCriticNetMultiHead(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks):
        super(ActorCriticNetMultiHead, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body

        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        for i in range(num_tasks):
            setattr(self, 'fc_action_head_{0}'.format(i), \
                layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3))
            setattr(self, 'fc_critic_head_{0}'.format(i), \
                layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3))

        self.actor_params = list(self.actor_body.parameters())
        self.critic_params = list(self.critic_body.parameters())
        self.phi_params = list(self.phi_body.parameters())

        for i in range(num_tasks):
            self.actor_params += list((getattr(self, 'fc_action_head_{0}'.format(i))).parameters())
            self.critic_params += list((getattr(self, 'fc_critic_head_{0}'.format(i))).parameters())

class CategoricalActorCriticNet_CL_MultiHead(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 num_tasks,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet_CL_MultiHead, self).__init__()
        self.network = ActorCriticNetMultiHead(state_dim, action_dim, phi_body, actor_body, \
            critic_body, num_tasks)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)
        self.task_mapper = []
        from ..shell_modules.mmn.ssmask_utils import set_model_task
        self.set_model_task = set_model_task

    def _task_label_to_id(self, task_label):
        for idx, _task_label in enumerate(self.task_mapper):
            if torch.equal(task_label, _task_label):
                return idx
        self.task_mapper.append(task_label)
        task_id = len(self.task_mapper) - 1
        print('task_label:', task_label)
        print('mapper:', self.task_mapper)
        print('task_id:', task_id)
        return task_id
        

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)

        try:
            if len(task_label)>1:
                task_id = self._task_label_to_id(task_label)
            elif type(task_label) == torch.Tensor:
                task_id = int(task_label.item())
            elif type(task_label) == np.ndarray:
                task_id = int(task_label[0])
        except TypeError:
            if type(task_label) == int:
                task_id = task_label
            elif type(task_label) == torch.Tensor:
                task_id = int(task_label.item())
            elif type(task_label) == np.ndarray:
                task_id = int(task_label[0])
            else:
                raise Exception("Type not recognised")
        self.set_model_task(self.network, task_id)
        phi, out = self.network.phi_body(obs)
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')

        _actor = getattr(self.network, 'fc_action_head_{0}'.format(task_id))
        _critic = getattr(self.network, 'fc_critic_head_{0}'.format(task_id))
        logits = _actor(phi_a)
        v = _critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            pass
            # layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, phi
