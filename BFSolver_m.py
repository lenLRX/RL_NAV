#coding=utf-8
'''
"暴力求解器"
每次遍历整个地图，看能不能走到相邻的格子
'''
import env

import matplotlib.pyplot as plt
import seaborn as sns

import heapq
import numpy as np

import csv
import math
import copy

#用来标识一个方向永远不让走了，除非重置这个节点
#比如A节点9点到过飞机，右边的节点一直风大，10点A点风也大了
#如果允许A节点11点飞过去，那就相当于在A点悬停到11点，实际应该要坠毁的！
INVALID = -2

class latticeObj(object):
    def __init__(self, loc):
        self.loc = loc
        self.reset()
    
    def reset(self):
        self.from_loc = None
        self.t = {
            (self.loc[0] - 1, self.loc[1]) : -1,
            (self.loc[0] + 1, self.loc[1]) : -1,
            (self.loc[0], self.loc[1] - 1) : -1,
            (self.loc[0], self.loc[1] + 1) : -1
        }
        self.leave_t = {
            (self.loc[0] - 1, self.loc[1]) : -1,
            (self.loc[0] + 1, self.loc[1]) : -1,
            (self.loc[0], self.loc[1] - 1) : -1,
            (self.loc[0], self.loc[1] + 1) : -1
        }

    def __str__(self):
        return "loc(%d,%d) t=%d"%(self.loc[0], self.loc[1], self.t)

    def __repr__(self):
        return "loc(%d,%d) t=%d"%(self.loc[0], self.loc[1], self.t)


class BFSolver(object):
    def __init__(self, env, dest):
        self.t = 0
        self.env = env
        self.init_lattice()
        self.src = self.env.start_location
        self.dest = dest

        #直接起飞，延迟起飞由“坠毁重置”模拟
        self.lattice[self.src[0]][self.src[1]].t = 0

    
    def init_lattice(self):
        self.lattice = []
        for i in range(self.env.dims[0]):
            row = []
            for j in range(self.env.dims[1]):
                row.append(latticeObj((i,j)))
            self.lattice.append(row)

    #当前时间这个地方的风速
    def get_windspeed(self, loc):
        return self.env.datas[self.t // self.env.tick_per_hour][loc[0]][loc[1]]

    #尝试是否能飞到相邻的格子
    def test_loc(self, loc, from_loc):
        obj = self.lattice[loc[0]][loc[1]]
        #这个地方还没从这个方向来过，并且没有被禁用
        if obj.t[from_loc] < 0 and self.get_windspeed(loc) < 15.0 and from_obj.leave_t[obj.loc] != INVALID:
            obj.t[from_loc] = self.t + 1
            obj.from_loc = from_loc

            from_obj = self.lattice[from_loc[0]][from_loc[1]]
            from_obj.leave_t[obj.loc] = obj.t
    
    #每到一个整点，风速变了，需要把所有风速超过15地方的飞机打下来
    def crash_reset(self):
        for i in range(self.env.dims[0]):
            for j in range(self.env.dims[1]):
                obj = self.lattice[loc[0]][loc[1]]
                if self.get_windspeed(loc) >= 15.0 
                    for origin in obj.t:
                        #在这一时刻飞过来的方向都需要回溯
                        if obj.t[origin] == self.t:
                            obj.t[origin]= -1

                            #来源
                            from_obj = self.lattice[origin[0]][origin[1]]
                            #上游
                            from_obj.leave_t[obj.loc] = -1
                            #如果上游风也大
                            while self.get_windspeed(from_obj.loc) >= 15.0:
                                #如果from_obj没有离开的路径,说明它正在悬停，坠毁！
                                out_num = 0
                                for k in from_obj.leave_t:
                                    if from_obj.leave_t[k] >= 0:
                                        out_num = out_num + 1
                                    else:
                                        #这个方向之后都不允许飞了
                                        from_obj.leave_t[k] = INVALID
                                
                                prev_from_obj = from_obj.from_obj
                                #如果这里坠毁，继续向上回溯
                                if out_num == 0:
                                    #需要重置此节点状态，之后还是可以飞过来的
                                    from_obj.reset()
                                    prev_from_obj.pop(from_obj)
                                    from_obj = prev_from_obj
                                else:
                                    break

    def solve(self):
        for self.t in range(0,self.env.total_hours * self.env.tick_per_hour):
            print(self.t)
            for i in range(self.env.dims[0]):
                for j in range(self.env.dims[1]):
                    #除了还没到过的地方和下一时刻才到的地方
                    if self.lattice[i][j].t < 0 or self.lattice[i][j].t > self.t:
                        continue
                    #left
                    if i - 1 >= 0:
                        self.test_loc((i - 1, j), (i, j))
                    #right
                    if i + 1 < self.env.dims[0]:
                        self.test_loc((i + 1, j), (i, j))
                    #up
                    if j - 1 >= 0:
                        self.test_loc((i, j - 1), (i, j))
                    #down
                    if j + 1 < self.env.dims[1]:
                        self.test_loc((i, j + 1), (i, j))
            
        print(self.get_score())
    
    def get_score(self):
        for dest in self.env.targets:
            print(self.lattice[dest[0]][dest[1]])
        return self.lattice[self.dest[0]][self.dest[1]]

    def dump(self):
        with open("result.csv", 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(10):
                loc = self.env.targets[i]
                
                if loc is None:
                    continue
                
                #到了终点
                if self.lattice[loc[0]][loc[1]].t > 0:
                    path = []
                    while True:
                        obj = self.lattice[loc[0]][loc[1]]
                        path.append(obj)
                        loc = obj.from_loc
                        if loc is None:
                            break
                
                    fwd_path = list(reversed(path))

                    #终点不太一样，它的离开时间对于输出路径没有意义
                    for idx in range(len(fwd_path) - 1):
                        obj = fwd_path[idx]
                        #从到达到离开的时间都要输出
                        #需要指定具体去哪的时间
                        for t in range(obj.t, obj.leave_t[fwd_path[idx + 1].loc]):
                            writer.writerow([i + 1, self.env.day,
                                '%02d:%02d'%(t // self.env.tick_per_hour + 3, t % self.env.tick_per_hour), obj.loc[0] + 1, obj.loc[1] + 1])
                    #终点再输出一行
                    fin = fwd_path[-1]
                    writer.writerow([i + 1, self.env.day,
                        '%02d:%02d'%(fin.t // self.env.tick_per_hour + 3, fin.t % self.env.tick_per_hour), fin.loc[0] + 1, fin.loc[1] + 1])

    def draw_path(self):
        color={
        0:       '#FFFF00',
        1:       '#FF6347',
        2:       '#2E8B57',
        3:       '#FAA460',
        4:       '#EE82EE',
        5:       '#008080',
        6:       '#4169E1',
        7:       '#800080',
        8:       '#48D1CC',
        9:       '#FF0000'}
        plt.subplots(figsize=(25,20))
        for i in range(10):
            loc = self.env.targets[i]
            if loc is None:
                continue
            if self.lattice[loc[0]][loc[1]].t > 0:
                while True:
                    plt.scatter(x=[loc[0]],y=[loc[1]],c=color[i])
                    loc = self.lattice[loc[0]][loc[1]].from_loc
                    if loc is None:
                        break
        plt.scatter(x=[item[0] for item in self.env.targets ],y=[item[1] for item in self.env.targets ],c='g')
        plt.grid(True,color='g',linestyle='--',linewidth='1')
        plt.savefig('BFpath.png',dpi=100)
        plt.show()

def main(env):
    solver = BFSolver(env,env.targets[0])
    solver.solve()

if __name__ == '__main__':
    myenv = env.Env('tbl_TrueData4Test',6)
    main(myenv)