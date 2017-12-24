#coding=utf-8
'''
“实时”求解器，即每次计算距离时用的是“当时”的风速
"当时"指的是从起点走到该点所需要的时间
时间很简单，就是dx+dy，因为你一次只能横着或竖着走一个格子
算法：
一开始整个地图格子填上INF的距离，起点距离为0
然后有个队列优先队列Q->[起点]
每次从队列头取出一个格子，计算周围四个格子到起点的”实时“距离，
如果这个格子现在据起点的距离比它记录的距离要短，那么就更新它的距离，并且将这个格子放入队列中
当然，为了记录路径，我们还需要记录每个节点的最短记录是从哪走过来了的
重复以上过程直到Q为空。

为了解决起飞的问题，我们可以每个小时都尝试一次（暴力）
取最短时间
'''
import env

import heapq
import numpy as np

import math
import copy

class latticeObj(object):
    def __init__(self, loc, distance):
        self.loc = loc
        self.distance = distance
        self.from_loc = None
    
    def __le__(self, other):
        return self.distance <= other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __str__(self):
        return "loc(%d,%d) distance=%f"%(self.loc[0], self.loc[1], self.distance)

    def __repr__(self):
        return "loc(%d,%d) distance=%f"%(self.loc[0], self.loc[1], self.distance)


class RTSolver(object):
    def __init__(self, env, dest, starthour):
        self.t = starthour
        self.env = env
        self.init_lattice()
        self.open_set = set()
        self.Q = []
        self.src = self.env.start_location
        self.dest = dest

        if self.src[0] > self.dest[0]:
            self.x_dir = -1
        else:
            self.x_dir = 1
        
        if self.src[1] > self.dest[1]:
            self.y_dir = -1
        else:
            self.y_dir = 1

        #init src
        loc = self.src
        start_obj = self.lattice[loc[0]][loc[1]]
        start_obj.distance = 0
        heapq.heappush(self.Q, start_obj)
        self.open_set.add(loc)
    
    def init_lattice(self):
        self.lattice = []
        for i in range(self.env.dims[0]):
            row = []
            for j in range(self.env.dims[1]):
                row.append(latticeObj((i,j),np.inf))
            self.lattice.append(row)
    
    def pop_from_Q(self):
        ret = self.Q[0]
        self.open_set.remove(ret.loc)
        heapq.heappop(self.Q)
        return ret

    #当前时间
    def get_time(self, distance):
        return self.t + distance // self.env.tick_per_hour

    #当前时间这个地方的风速
    def get_windspeed(self, loc, dist):
        return self.env.datas[self.get_time(dist)][loc[0]][loc[1]]

    def test_loc(self, loc, dist, from_loc):
        obj = self.lattice[loc[0]][loc[1]]
        #print(obj)
        #找到更近的路，而且现在风速 < 15
        if dist < obj.distance and self.get_windspeed(loc, dist) < 15.0:
            obj.distance = dist
            obj.from_loc = from_loc
            #这个地点不在队列中，入队
            if not obj.loc in self.open_set:
                heapq.heappush(self.Q, obj)
                #记录一下路径，方便之后回溯
                self.open_set.add(loc)
            #这个地点在队列中，但是距离变短了，重新排序
            else:
                heapq.heapify(self.Q)

    def solve(self):
        while len(self.Q) > 0:
            #print(self.Q)
            p = self.pop_from_Q()
            print(p)
            loc = p.loc
            nextdist = p.distance + 1
            if self.get_time(nextdist) >= self.env.total_hours:
                continue
            #left
            if loc[0] - 1 >= 0:
                self.test_loc((loc[0] - 1, loc[1]), nextdist, loc)
            #right
            if loc[0] + 1 < self.env.dims[0]:
                self.test_loc((loc[0] + 1, loc[1]), nextdist, loc)
            #up
            if loc[1] - 1 >= 0:
                self.test_loc((loc[0], loc[1] - 1), nextdist, loc)
            #down
            if loc[1] + 1 < self.env.dims[1]:
                self.test_loc((loc[0], loc[1] + 1), nextdist, loc)
            
        print(self.get_score())
    
    def get_score(self):
        return self.lattice[self.dest[0]][self.dest[1]].distance

def main(env):
    solver = RTSolver(env,env.targets[0],0)
    solver.solve()

if __name__ == '__main__':
    myenv = env.Env('tbl_TrueData4Test',6)
    main(myenv)