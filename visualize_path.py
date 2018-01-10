import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from matplotlib.pyplot import cm


windData = np.zeros((548, 421,5,18))
X = 548
Y = 421
DATE = 5
HOUR = 18
MODULE = 10
PATH3 = np.zeros((5,18,548,421))

def load_data(windFilePath,resultFilePath):
    with open(windFilePath,newline='') as csvfile:
        testreader = csv.reader(csvfile)
        row_num = 0
        ###pack = np.zeros((5, 548,421,18,10))
        m = 0
        for row in testreader:
            row_num = row_num+1
            if row_num == 1:
                continue
            '''
            x_id = int(row[-5])-1
            y_id = int(row[-4])-1
            date = int(row[-3])-6
            hour = int(row[-2])-3
            ###wind = float(row[-1])
            s = row[-1].split('[torch')
            s = s[0]
            ###s = s.strip()
            s = s.replace('\n','')
            if 'e' in s:
                wind = 0
            else:
                wind = float(s)

            if wind < 1.0:
                windData[x_id][y_id][date][hour]=wind###区分风速大于或小于15
            else:
                windData[x_id][y_id][date][hour]=wind+2
            if row_num == 2:
                print(x_id,y_id,date,hour,wind)
            '''

            
            x_id = int(row[-6])-1
            y_id = int(row[-5])-1
            date = int(row[-4])-6
            hour = int(row[-3])-3
            m = m + float(row[-1])
            if (row_num-1)%10 == 0:
                wind = m/10
                wind =  wind/16
                m = 0
                if wind < 1.0:
                    windData[x_id][y_id][date][hour]=wind###区分风速大于或小于15
                else:
                    windData[x_id][y_id][date][hour]=wind+2
            if row_num == 2:
                print(x_id,y_id,date,hour,m)

                        


    
    with open(resultFilePath,newline='') as csvfile:
        testreader = csv.reader(csvfile)
        for row in testreader:
            x_id = int(row[-2])-1
            y_id = int(row[-1])-1
            t   =  row[-3]
            t = t.split(':')
            hour = int(t[0])-3
            minute = int(t[1])
            date = int(row[-4])-6
            dest = int(row[-5])  
            PATH3[date][hour][x_id][y_id]=1
                
            

def checkData(wfp,rfp):
    load_data(wfp,rfp)
    for d in range(DATE-1):
        s = sum(sum(sum(PATH3[d]-PATH3[d+1])))
    print(s)


def drawWindPic(windFpath,resultFpath):
    ###load_data(windFpath,resultFpath)
    pack = []
    color = []
    for d in range(DATE):
        for h in range(HOUR):
            for y in range(Y):
                for x in range(X):
                    if PATH3[d][h][x][y] == 1:
                        pack.append(20.0)
                    else:
                        pack.append(windData[x][y][d][h])
                _itm = np.asarray(pack)
                color.append(_itm)
                pack = []
            _itm = np.asarray(color)
            color = []
            norm = plt.Normalize(vmin=_itm.min(), vmax=_itm.max())
            ###plt.imshow(_itm, interpolation='none', cmap='binary', origin='lower')
            
            ##plt.imsave('pathPic/'+str(d)+'_'+str(h)+'.png') 
            ###cmap = plt.cm.colors.Colormap.is_gray()
            ###image = cmap(norm(data))
            plt.imsave(arr = _itm,fname = 'pathPic/'+str(d)+'_'+str(h)+'.png',format = 'png',origin = 'lower')
            print('pic '+str(d)+'_'+str(h)+'.png'+' saved')

wfpath = './pred.csv'
resultFpath = './result.csv'
windTestPath = '../tianchi/data/ForecastDataforTesting_201712.csv'
if __name__ == '__main__':
    checkData(windTestPath,resultFpath)
    drawWindPic(windTestPath,resultFpath)
