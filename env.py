import numpy as np

import os,csv

import dbconn

class InvalidActionException(Exception):
    pass

class Env(object):
    def __init__(self, tbl, day, modelno = None):
        self.dims = [548,421]
        self.load_targets()
        self.table_name = tbl
        self.day = day
        self.modelno = modelno
        self.hours = range(3, 21)
        self.total_hours = 18
        self.datas = []
        self.log = []
        #30 tick per hour
        self.tick_per_hour = 30
        self.time = 0
        self.score = 0
        #10 city to visit
        self.remain_task = 10

        self.action_orders = []
        self.done = []
        self.trajectories = []
        self.start_time = []
        self.scores = []
        for i in range(10):
            self.done.append(False)
            self.action_orders.append(None)
            self.locs.append(self.start_location)
            self.trajectories.append([])
            self.start_time.append(-1)
            self.scores.append(0)
        
        

        conn = dbconn.get_conn()

        for hour in self.hours:
            self.datas.append(self.get_one_hour(conn, self.day, hour))
        
        
    
    def windspeed_at(self, loc):
        return self.datas[self.get_curr_hour()][loc[0]][loc[1]]

    def windspeed_at_hour(self, hour, loc):
        return self.datas[hour][loc[0]][loc[1]]
    
    def set_done(self, i):
        self.done[i] = True
        self.remain_task = self.remain_task - 1

    def if_reach_target_or_die(self, i):
        if self.locs[i] == self.targets[i]:
            print('%d reach target!'%i)
            self.set_done(i)
            _score = self.time * 2
            self.score = self.score + _score
            self.scores[i] = _score
            return 1
        if self.windspeed_at(self.locs[i]) >= 15:
            print('%d crashed!'%i)
            self.set_done(i)
            _score = 24 * 60
            self.score = self.score + _score
            self.scores[i] = _score
            return -1

        return 0

    def tick(self):
        m_reward = 0.01
        rewards = []
        for i in range(10):
            if not self.done[i]:
                self.trajectories[i].append(self.locs[i])
                act = self.action_orders[i]
                #does not take off yet
                if act is None:
                    rewards.append(0.0)
                    continue
                if abs(act[0]) + abs(act[1]) > 1:
                    raise InvalidActionException()
                if self.start_time[i] < 0:
                    #record the take off time
                    self.start_time[i] = self.time
                _loc = self.locs[i]
                self.locs[i] = (_loc[0] + act[0], _loc[1] + act[1])
                ret = self.if_reach_target_or_die(i)
                target = self.targets[i]
                if 0 == ret:
                    if act[0] > 0:
                        if _loc[0] >= target[0]:
                            ret = -m_reward
                        elif _loc[0] < target[0]:
                            ret = m_reward
                    elif act[0] < 0:
                        if _loc[0] <= target[0]:
                            ret = -m_reward
                        elif _loc[0] > target[0]:
                            ret = m_reward
                    elif act[1] > 0:
                        if _loc[1] >= target[1]:
                            ret = -m_reward
                        elif _loc[1] < target[1]:
                            ret = m_reward
                    elif act[1] < 0:
                        if _loc[1] <= target[1]:
                            ret = -m_reward
                        elif _loc[1] > target[1]:
                            ret = m_reward
                rewards.append(ret)
            else:
                rewards.append(None)
        self.time = self.time + 1
        print("t:%d rewards %s"%(self.time,str(rewards)))
        return rewards
    
    def load_targets(self):
        self.targets = []
        self.locs = []
        with open(os.path.join('data', 'CityData.csv'), newline='') as csvfile:
            cityreader = csv.reader(csvfile)
            for lineno,row in enumerate(cityreader):
                if lineno == 1:
                    loc = (int(row[1]) - 1, int(row[2]) - 1)
                    self.start_location = loc
                elif lineno > 1:
                    loc = (int(row[1]) - 1, int(row[2]) - 1)
                    self.targets.append(loc)
    
    def set_move(self, i, m):
        self.action_orders[i] = m

    def get_curr_hour(self):
        return self.time // self.tick_per_hour

    def get_one_hour(self, conn, day, hour, modelno = None):
        h = np.zeros((548,421))
        d = dbconn.query_data_by_table(conn, self.table_name, day, hour, modelno)
        for row in d:
            h[row[0] - 1][row[1] - 1] = row[-1]
        return h

    def end(self):
        return self.time >= self.tick_per_hour * self.total_hours \
            or self.remain_task == 0
    
    def dump(self, path = 'result.csv'):
        with open(path, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(10):
                t = self.start_time[i]
                print(self.trajectories[i])
                for loc in self.trajectories[i]:
                    writer.writerow([i + 1, self.day,
                        '%02d:%02d'%(t // self.tick_per_hour + 3, t % self.tick_per_hour),loc[0] + 1, loc[1] + 1])
                    t = t + 1
                writer.writerow([i + 1, self.day,
                        '%02d:%02d'%(t // self.tick_per_hour + 3, t % self.tick_per_hour), self.targets[i][0] + 1, self.targets[i][1] + 1])