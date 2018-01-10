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


fpath = '../tianchi/data/ForecastDataforTesting_201712.csv'
frealpath = '../tianchi/data/In_situMeasurementforTraining_201712.csv'

Data = np.zeros((5, 548,421,18,10))
X = 548
Y = 421
DATE = 5
HOUR = 18
MODULE = 10
upd_batch = 1000

def load_data(fpath):
    with open(fpath,newline='') as csvfile:
        testreader = csv.reader(csvfile)
        row_num = 0
        ###pack = np.zeros((5, 548,421,18,10))
        pack = []
        for row in testreader:
            row_num = row_num+1
            if row_num == 1:
                continue
            x_id = int(row[-6])-1
            y_id = int(row[-5])-1
            date = int(row[-4])-1
            hour = int(row[-3])-3
            model = int(row[-2])-1
            wind = float(row[-1])
            ####Data[date][x_id][y_id][hour][model] = wind
            if date not in pack:
                pack.append(date)
                print(date)
        
                

def loading_origin_fn(fpath, buffer, mutex):
    load_data(fpath)
    batch = []
    pack = []
    BIAS = [-1,0,1]
    for d in range(DATE):
        for h in range(HOUR):
            for x in range(X):
                for y in range(Y):
                    for xbias in BIAS:
                        for ybias in BIAS:
                            if x+xbias < X and y+ybias < Y and x+xbias >=0 and y+ybias >=0 and h-1 >= 0:
                                for m in range(MODULE):
                                    pack.append(Data[d][x+xbias][y+ybias][h-1][m])###如果附近的块数据存在，就把附近的数据加进去
                            elif h-1 >= 0:
                                for m in range(MODULE):
                                    pack.append(Data[d][x][y][h-1][m])###如果附近的数据不存在，就把其它自己的数据加进去，作补充，保证数据维度一致
                            else:
                                for m in range(MODULE):
                                    pack.append(Data[d][x][y][h][m])
                    pack.append((h+3)/24.0)
                    pack.append(float(x/X))
                    pack.append(float(y/Y))
                    _itm = np.asarray(pack)
                    batch.append(_itm)
                    pack=[]
                    if len(batch) >= upd_batch:
                        with mutex:
                            buffer.extend(batch)
                        batch = []
    with mutex:
        buffer.extend(batch)
    print("data done %d"%(len(buffer)))


if __name__ == '__main__':
    load_data(fpath)
