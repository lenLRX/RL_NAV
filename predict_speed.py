import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch import optim

import threading

import os
import csv

import time

tensor_type = torch.FloatTensor

use_cuda = True and torch.cuda.is_available()

if use_cuda:
    tensor_type = torch.cuda.FloatTensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.in_dim = 13
        self.h_dim = 20
        self.layer_num = 5
        self.out_dim = 1

        
        #input layer
        self.input_layer = nn.Linear(self.in_dim, self.h_dim)
        self.init_layer(self.input_layer)
        self.input_layer_bn = nn.BatchNorm1d(self.h_dim)

        self.hidden_layers = []
        self.hidden_bns = []

        for i in range(self.layer_num):
            _layer = nn.Linear(self.h_dim, self.h_dim)
            setattr(self, "hidden_layer_%d"%i, _layer)
            self.hidden_layers.append(_layer)
            self.init_layer(_layer)
            _layer_bn = nn.BatchNorm1d(self.h_dim)
            setattr(self, "hidden_bn_%d"%i, _layer_bn)
            self.hidden_bns.append(_layer_bn)


        self.output_layer = nn.Linear(self.h_dim, self.out_dim)
        self.init_layer(self.output_layer)
        

        # mode
        self.train()
    
    def init_layer(self,layer):
        init.xavier_uniform(layer.weight, gain=np.sqrt(2))
        init.constant(layer.bias, 0.01)

    def forward(self,inputs):
        f = F.sigmoid
        h = f(self.input_layer_bn(self.input_layer(inputs)))
        for i in range(self.layer_num):
            h = self.hidden_layers[i](h)
            h = f(self.hidden_bns[i](h))
        out = self.output_layer(h)
        return out

upd_batch = 10000

def loading_origin_fn(fpath, buffer, mutex):
    with open(fpath, newline='') as csvfile:
        testreader = csv.reader(csvfile)
        row_num = 0
        batch = []
        pack = []
        for row in testreader:
            row_num = row_num + 1
            if row_num == 1:
                continue
            #print(row)
            hour = float(row[-3])
            pack.append(float(row[-1]) / 15.0)
            #10 row a pack
            if 0 == (row_num - 1) % 10:
                pack.append(float(row[0]) / 548.0)
                pack.append(float(row[0]) / 421.0)
                pack.append(hour / 24.0)
                _itm = np.asarray(pack)
                batch.append(_itm)
                pack = []
            if len(batch) >= upd_batch:
                with mutex:
                    buffer.extend(batch)
                batch = []
        with mutex:
            buffer.extend(batch)
        print("data done %d,%d"%(len(buffer),row_num))

def loading_true_fn(fpath, buffer, mutex):
    with open(fpath, newline='') as csvfile:
        testreader = csv.reader(csvfile)
        row_num = 0
        batch = []
        for row in testreader:
            row_num = row_num + 1
            if row_num == 1:
                continue
            batch.append(float(row[-1]) / 15.0)
            if len(batch) >= upd_batch:
                with mutex:
                    buffer.extend(batch)
                batch = []
        with mutex:
            buffer.extend(batch)
        
        print("label done %d,%d"%(len(buffer),row_num))


def shuffle_fn(buffer, mutex):
    return
    while True:
        time.sleep(1000)
        with mutex:
            np.random.shuffle(buffer)
            print('shuffle!')

class DataProvider(object):
    '''
        Async Data Provider
    '''
    def __init__(self, fpath, frealpath):
        '''
        starts the loading thread and shuffle thread
        '''
        self.buffer = []
        self.real_data = []
        self.idx = []
        self.load_finished = False
        self.buffer_mtx = threading.Lock()
        self.loading_t = threading.Thread(target=loading_origin_fn, args=(fpath, self.buffer, self.buffer_mtx))
        self.loading_true_t = threading.Thread(target=loading_true_fn, args=(frealpath, self.real_data, self.buffer_mtx))
        self.shuffle_fn_t = threading.Thread(target=shuffle_fn,args=(self.idx, self.buffer_mtx))

        self.loading_true_t.start()
        self.loading_t.start()
        self.shuffle_fn_t.start()

        threading.Thread(target=self.wait_finish,args=()).start()
    
    def wait_finish(self):
        self.loading_true_t.join()
        self.loading_t.join()
        with self.buffer_mtx:
            self.load_finished = True
            self.idx = [x for x in range(len(self.buffer))]
        
        print('loading finished')

    def get(self, batch_size):
        while True:
            with self.buffer_mtx:
                buffer_len = min(len(self.buffer), len(self.real_data))
                #print(buffer_len)
                if buffer_len < batch_size:
                    continue
                self.idx = np.random.randint(buffer_len, size=batch_size)                
                pred_data = [self.buffer[i] for i in self.idx]
                real_data = [self.real_data[i] for i in self.idx]
                return np.vstack(pred_data), np.asarray(real_data)
            time.sleep(1)
    

def training_task():
    #fpath = os.path.join('data', 'ForecastDataforTraining_20171205', 'ForecastDataforTraining_201712.csv')
    #frealpath = os.path.join('data', 'In_situMeasurementforTraining_20171205', 'In_situMeasurementforTraining_201712.csv')
    fpath = './data/ForecastDataforTraining_20171205/ForecastDataforTraining_201712.csv'
    frealpath = './data/In_situMeasurementforTraining_20171205/In_situMeasurementforTraining_201712.csv'
    provider = DataProvider(fpath,frealpath)

    model = Model()
    if use_cuda:
        model = model.cuda()
        print('using cuda')
    else:
        print('using cpu')
    optimizer = optim.Adam(model.parameters(), lr = 1E-3)
    while True:
        optimizer.zero_grad()
        data_, label_ = provider.get(5000)
        t1 = time.time()
        var_label_ = Variable(tensor_type(label_))
        var_data_ = Variable(tensor_type(data_))
        out_ = model(var_data_)
        loss = (out_ - var_label_) ** 2
        loss = torch.mean(loss)
        print(loss.data[0], out_.data[0][0], label_.data[0])
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(),'saved_model')


if __name__ == '__main__':
    training_task()
