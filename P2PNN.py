import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn.init as init

from torch import optim

import pandas as pd

import math

Data = np.zeros((5,18,548,421,10))
TruthData = np.zeros((5,18,548,421))

X = 548
Y = 421
DATE = 5
HOUR = 18
MODULE = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        #input layer
        self.conv_layer1 = nn.Conv2d(10, 10, 3)
        self.conv_layer2 = nn.Conv2d(10, 10, 3)
        self.conv_layer3 = nn.Conv2d(10, 10, 3)
        self.conv_layer4 = nn.Conv2d(10, 10, 3)
        #1x1x6x5 now

        self.up_sample1 = nn.ConvTranspose2d(10, 5, 3)
        self.up_sample2 = nn.ConvTranspose2d(5, 3, 3)
        self.up_sample3 = nn.ConvTranspose2d(3, 1, 3)
        self.up_sample4 = nn.ConvTranspose2d(1, 1, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # mode
        self.train()
    
    def init_layer(self,layer):
        init.xavier_uniform(layer.weight, gain=np.sqrt(2))
        init.constant(layer.bias, 0.01)

    def forward(self,inputs):
        origin_input = Variable(torch.from_numpy(np.asarray([inputs])).float())
        in_var = F.sigmoid(self.conv_layer1(origin_input))
        in_var = F.sigmoid(self.conv_layer2(in_var))
        in_var = F.sigmoid(self.conv_layer3(in_var))
        in_var = F.sigmoid(self.conv_layer4(in_var))

        in_var = F.sigmoid(self.up_sample1(in_var))
        in_var = F.sigmoid(self.up_sample2(in_var))
        in_var = F.sigmoid(self.up_sample3(in_var))
        in_var = F.relu(self.up_sample4(in_var, output_size = origin_input.size()))

        return in_var


filePath = './data/ForecastDataforTraining_20171205/ForecastDataforTraining_201712.csv'
tfilePath = './data/In_situMeasurementforTraining_20171205/In_situMeasurementforTraining_201712.csv'


def training_process(Data, TruthData):
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr = 1E-2)
    count = 0
    while True:
        dates = list(range(DATE))
        hours = list(range(HOUR))
        count = count + 1
        for d in dates:
            for h in range(HOUR):
                optimizer.zero_grad()
                pred = model(Data[d][h] / 15.0).view((X,Y))
                trueData = Variable(torch.from_numpy(np.asarray(TruthData[d][h] / 15.0)).float())
                loss = pred - trueData
                loss = loss ** 2
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                print(loss)
        
        if count % 100 == 0:
            torch.save(model.state_dict(),'./model/saved_model_%d'%count)

def load_data():
    Data = pd.read_csv(filePath)
    Data = Data['wind'].values.reshape((DATE,HOUR,X,Y,MODULE))
    # x => m
    Data = np.swapaxes(Data,2,4)
    # x => y
    Data = np.swapaxes(Data,3,4)
    TruthData = pd.read_csv(tfilePath)
    TruthData = TruthData['wind'].values.reshape((DATE,HOUR,X,Y))

    print('load done')

    return Data, TruthData

if __name__ == "__main__":
    Data, TruthData = load_data()
    training_process(Data, TruthData)