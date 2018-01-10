import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from matplotlib.pyplot import cm

'''''
这里是把训练数据的十个模型与真值做差，看看哪个模型比较好
'''''
Data = np.zeros((5,18,548,421,10))
TruthData = np.zeros((5,18,548,421))

X = 548
Y = 421
DATE = 5
HOUR = 18
MODULE = 10


def load_data(filePath,tfilePath):
    with open(filePath,newline='') as csvfile:
        testreader = csv.reader(csvfile)
        row_num = 0

        for row in testreader:
            row_num = row_num + 1
            if row_num == 1:
                continue
            x_id = int(row[-6])-1
            y_id = int(row[-5])-1
            date = int(row[-4])-3
            hour = int(row[-3])-3
            mode = int(row[-2])-1
            wind = float(row[-1])

            Data[date][hour][x_id][y_id][mode] = wind
    
    with open(tfilePath,newline='') as csvfile:
        testreader = csv.reader(csvfile)
        row_num = 0

        for row in testreader:
            row_num = row_num + 1
            if row_num == 1:
                continue
            x_id = int(row[-5])-1
            y_id = int(row[-4])-1
            date = int(row[-3])-3
            hour = int(row[-2])-3
            wind = float(row[-1])

            TruthData[date][hour][x_id][y_id] = wind

def drawDifference(filePath,tfilePath):
    load_data(filePath,tfilePath)
    
    differ = np.zeros((X,Y))

    for d in range(DATE):
        for h in range(HOUR):
            for m in range(MODULE):
                sum = 0
                max = 0
                for x in range(X):
                    for y in range(Y):
                        differ[x][y] = Data[d][h][x][y][m]-TruthData[d][h][x][y]
                        sum = sum +abs(differ[x][y])
                        if max < abs(differ[x][y]):
                            max = abs(differ[x][y])
                ave = sum/(X*Y)
                plt.imsave(arr = differ,fname = '../data/modelPic/'+str(d)+'_'+str(h)+'_'+str(m)+'.png',format = 'png',origin = 'lower')
                print('pic '+str(d)+'_'+str(h)+'_'+str(m)+'.png'+' saved')
                print('sum differ = '+str(sum)+'\nave differ = '+str(ave)+'\nmax differ = '+str(max)+'\n for pic '+str(d)+'_'+str(h)+'_'+str(m)+'.png')

filePath = '../data/ForecastDataforTraining_201712.csv'
tfilePath = '../data/In_situMeasurementforTraining_201712.csv'

if __name__ == '__main__':
    drawDifference(filePath,tfilePath)