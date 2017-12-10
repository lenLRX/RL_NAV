import env


def test():
    myenv = env.Env('tbl_TrueData',1)
    while not myenv.end():
        for i in range(10):
            if not myenv.done[i]:
                loc = myenv.locs[i]
                target = myenv.targets[i]
                x,y = (0,0)
                if loc[0] > target[0]:
                    x = -1
                elif loc[1] > target[1]:
                    y = -1
                elif loc[0] < target[0]:
                    x = 1
                elif loc[1] < target[1]:
                    y = 1
                
                myenv.set_move(i, (x,y))
        myenv.tick()
    #print('score %d time %d remain_task %d'%(myenv.score, myenv.time, myenv.remain_task))
    #import pdb
    #pdb.set_trace()

if __name__ == '__main__':
    test()