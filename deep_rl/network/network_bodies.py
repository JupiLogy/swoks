#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
import time

class CustomConvBody(nn.Module):
    def __init__(self, in_channels=1):
        super(CustomConvBody, self).__init__()
        self.feature_dim=16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 3, kernel_size = (6,7), stride=(1, 3)))
        self.conv2 = layer_init(nn.Conv2d(3, 5, kernel_size=(3,5), stride=(1, 2)))
        self.conv3 = layer_init(nn.Conv2d(5, 6, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(648, self.feature_dim-1))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        features_detached=self.fc4(y).cpu().detach().numpy()
        #print(y.shape)
        y = F.tanh(self.fc4(y))
        return y,features_detached


class NatureConvBody(nn.Module):
    def __init__(self, in_channels=1):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1))
        self.conv2 = layer_init(nn.Conv2d(4, 8, kernel_size=3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(256, self.feature_dim-1))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        features_detached=self.fc4(y).cpu().detach().numpy()
        #print(y.shape)
        y = F.tanh(self.fc4(y))
        return y,features_detached


from ..shell_modules.mmn.ssmask_utils import MultitaskMaskConv2d
from ..shell_modules.mmn.ssmask_utils import MultitaskMaskLinear
class NatureConvBodySS(nn.Module):
    def __init__(self, in_channels=1, num_tasks=8, discrete=True):
        super(NatureConvBodySS, self).__init__()
        self.feature_dim = 16
        self.conv1 = MultitaskMaskConv2d(in_channels, 2, kernel_size=5, stride=1, num_tasks=num_tasks, discrete=discrete)
        self.conv2 = MultitaskMaskConv2d(2, 4, kernel_size=3, stride=1, num_tasks=num_tasks, discrete=discrete)
        self.conv3 = MultitaskMaskConv2d(4, 8, kernel_size=3, stride=1, num_tasks=num_tasks, discrete=discrete)
        self.fc4 = MultitaskMaskLinear(128, self.feature_dim, num_tasks=num_tasks, discrete=discrete)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        features_detached=self.fc4(y).cpu().detach().numpy()
        y = F.tanh(self.fc4(y))
        return y,features_detached

class NatureConvBodySSmngrd(nn.Module):
    def __init__(self, in_channels=1, num_tasks=8, discrete=False):
        super(NatureConvBodySSmngrd, self).__init__()
        self.feature_dim = 200
        self.conv1 = MultitaskMaskConv2d(in_channels, 16, kernel_size=2, stride=1, num_tasks=num_tasks, discrete=discrete)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = MultitaskMaskConv2d(16, 32, kernel_size=2, stride=1, num_tasks=num_tasks, discrete=discrete)
        self.conv3 = MultitaskMaskConv2d(32, 64, kernel_size=2, stride=1, num_tasks=num_tasks, discrete=discrete)
        self.fc4 = MultitaskMaskLinear(64, self.feature_dim, num_tasks=num_tasks, discrete=discrete)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.maxp1(y)
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        features_detached=self.fc4(y).cpu().detach().numpy()
        y = F.tanh(self.fc4(y))
        return y,features_detached

class NatureConvBody_lstm(nn.Module):
    def __init__(self, in_channels=1):
        super(NatureConvBody_lstm, self).__init__()
        self.feature_dim = 16
        self.rnn_input_dim=256
        self.conv1 = layer_init(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1))
        self.conv2 = layer_init(nn.Conv2d(4, 8, kernel_size=3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=1))
        self.lstm = nn.LSTM(self.rnn_input_dim, self.feature_dim)
        self.fc4 = layer_init(nn.Linear(self.feature_dim, 16))
        self.hidden = self.init_hidden()
        self.reset_flag=False
        self.count=0

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, 1, self.feature_dim)).cuda(),
                torch.autograd.Variable(torch.zeros(1, 1, self.feature_dim)).cuda())

    def forward(self, x):
        self.count=self.count+1
        if self.reset_flag:
            self.hidden=self.repackage_hidden(self.hidden)
            self.reset_flag=False
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.tanh(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = y.view(-1, 1, self.rnn_input_dim)
        y, self.hidden = self.lstm(y, self.hidden)
        y = torch.squeeze(y, 1)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

    def repackage_hidden(self,h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

class CombinedNet(nn.Module, BaseNet):
    def __init__(self, bodyPredict):
        super(CombinedNet, self).__init__()
#        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
#        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.bodyPredict = bodyPredict
        self.to(Config.DEVICE)

    def returnFeatures(self, x, to_numpy=False):
        phi = self.bodyPredict(tensor(x))
        if to_numpy:
            return phi.cpu().detach().numpy()
        return phi
    #        value = self.fc_value(phi)
#        advantange = self.fc_advantage(phi)
#        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
#        if to_numpy:
#            return q.cpu().detach().numpy()
#        return q

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class DummyBody_CL(nn.Module):
    def __init__(self, state_dim, task_label_dim=None):
        super(DummyBody_CL, self).__init__()
        self.feature_dim = state_dim #+ (0 if task_label_dim is None else task_label_dim)
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        # if self.task_label_dim is not None:
        #     assert task_label is not None, '`task_label` should be set'
        #     x = torch.cat([x, task_label], dim=1)
        return x, []

class FCBody_CL(nn.Module): # fcbody for continual learning setup
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_CL, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + (task_label_dim,) ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        #if self.task_label_dim is not None:
        #    assert task_label is not None, '`task_label` should be set'
        #    import pdb; pdb.set_trace()
        #    x = torch.cat([x, task_label.repeat(8,1)], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)

        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for layer in self.layers:
                x = self.gate(layer(x))
        return x, ret_act
