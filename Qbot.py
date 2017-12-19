import env

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn.init as init

import math
import copy


move_map = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.h_size_1 = 20
        self.action_space = 5

        #input layer
        self.conv_layer1 = nn.Conv2d(18, 1, 3,stride = 3)
        self.conv_layer2 = nn.Conv2d(1, 1, 3,stride = 3)
        self.conv_layer3 = nn.Conv2d(1, 1, 3,stride = 3)
        self.conv_layer4 = nn.Conv2d(1, 1, 3,stride = 3)
        #1x1x6x5 now

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.value_layer = nn.Linear(6*5 + 2, 1)
        self.init_layer(self.value_layer)

        self.hidden_layer = nn.Linear(6*5 + 2, 5)
        self.init_layer(self.hidden_layer)

        # mode
        self.train()
        self.reset_conv()
    
    def reset_conv(self):
        self.conv = None
    
    def init_layer(self,layer):
        init.xavier_uniform(layer.weight, gain=np.sqrt(2))
        init.constant(layer.bias, 0.01)

    def init_lstm(self):
        self.lstm_hidden = (Variable(torch.zeros(1,1,self.h_size_2)),
                           Variable(torch.zeros(1,1,self.h_size_2)))

    def forward(self,inputs):
        windspeeds = inputs[0]
        pos = inputs[1]
        
        if self.conv is None:
            in_var = Variable(torch.from_numpy(np.asarray([windspeeds])).float())
            in_var = F.sigmoid(self.conv_layer1(in_var))
            in_var = F.sigmoid(self.conv_layer2(in_var))
            in_var = F.sigmoid(self.conv_layer3(in_var))
            in_var = F.sigmoid(self.conv_layer4(in_var))
            self.conv = in_var.view(-1)
        cat = torch.cat([self.conv, Variable(torch.FloatTensor(pos))])
        v = self.value_layer(cat)
        h = self.hidden_layer(cat)
        out = F.softmax(h)
        out_log = F.log_softmax(h)

        return out, out_log, v


class Agent(object):
    def __init__(self, model = None):
        if model is None:
            self.model = Model(None, None)
        
        self.discount_factor = 0.9
        self.reset()
    
    def reset(self):
        self.model.reset_conv()
        self.rewards = []
        self.actions = []
        self.act_logs = []
        self.values = []
    
    def forward(self, inputs):
        out, out_log, v = self.model(inputs)
        idx = np.argmax(out.data.numpy()[0])
        if np.random.rand(1)[0] < 0.05:
            idx = np.random.randint(5)
        self.actions.append(idx)
        self.act_logs.append(out_log)
        self.values.append(v)
        return idx

    def set_reward(self, reward):
        self.rewards.append(reward)
    
    def train(self):
        self.model.zero_grad()
        R = torch.zeros(1, 1)
        self.values.append(self.values[-1])

        R = Variable(R)
        A = Variable(torch.zeros(1, 1))    
        loss = Variable(torch.FloatTensor([0.0]))
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.discount_factor*R
            A = R - self.values[i].view(-1).data[0]
            _d_log = Variable(torch.zeros(1, self.act_logs[i].view(-1).size()[0]))
            _d_log.data[0][self.actions[i]] = 1
            _d_log = _d_log * self.act_logs[i]
            value_loss = (R - self.values[i].view(-1)) ** 2
            policy_loss = - (A * _d_log).view(-1)
            policy_loss = torch.mean(policy_loss)
            loss = loss + policy_loss + 0.5 * value_loss
        loss = loss / len(self.rewards)
        
        loss.backward()

        for n,p in self.model.named_parameters():
            p.data -= 1e-3 * p.grad.data
        
        print("loss", loss.data.numpy()[0][0])


def main(the_env):
    
    first = [True for x in range(10)]
    Agents = []
    for i in range(10):
        Agents.append(Agent())
    while True:
        for i in range(10):
            Agents[i].reset()
        myenv = copy.deepcopy(the_env)
        while not myenv.end():
            for i in range(10):
                if not myenv.done[i]:
                    loc = myenv.locs[i]
                    idx = Agents[i].forward((myenv.datas,(loc[0] / 548.0, loc[1] / 421.0)))
                    act = move_map[idx]

                    if first[i] and act == (0,0):
                        #do not take off
                        continue
                    else:
                        first[i] = False
                    
                    myenv.set_move(i, act)
            rewards = myenv.tick()
            for i in range(10):
                r = rewards[i]
                if not rewards[i] is None:
                    Agents[i].set_reward(rewards[i])
        
        for i in range(10):
            Agents[i].train()

        print('total score %d time %d remain_task %d'%(myenv.score, myenv.time, myenv.remain_task))
        #print('scores %s'%str(myenv.scores))

if __name__ == '__main__':
    myenv = env.Env('tbl_TrueData4Test',6)
    main(myenv)