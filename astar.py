import matplotlib.pyplot as plt
import seaborn as sns
import env
import heapq
class Cell():
    def __init__(self, x, y, reachable):
        """Initialize new cell.

        @param reachable is cell reachable? not a wall?
        @param x cell x coordinate
        @param y cell y coordinate
        @param g cost to move from the starting cell to this cell.
        @param h estimation of the cost to move from this cell
                 to the ending cell.
        @param f f = g + h
        """
        self.reachable = reachable
        self.x = x
        self.y = y
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0
    # <
    def __lt__(self,other):
        return self.f < other.f
class AStar():
    def __init__(self):
        # open list
        self.opened = []
        heapq.heapify(self.opened)
        # visited cells list
        self.closed = set()
        # grid cells
        self.cells = []
        self.grid_height = None
        self.grid_width = None

    def init_grid(self, width, height, walls, start, end):
        """Prepare grid cells, walls.

        @param width grid's width.
        @param height grid's height.
        @param walls list of wall x,y tuples.
        @param start grid starting point x,y tuple.
        @param end grid ending point x,y tuple.
        """
        self.grid_height = height
        self.grid_width = width
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if (x, y) in walls:
                    reachable = False
                else:
                    reachable = True
                self.cells.append(Cell(x, y, reachable))
        self.start = self.get_cell(*start)
        self.end = self.get_cell(*end)

    def get_heuristic(self, cell):
        """Compute the heuristic value H for a cell.

        Distance between this cell and the ending cell multiply by 10.

        @returns heuristic value H
        """
        return 10 * (abs(cell.x - self.end.x) + abs(cell.y - self.end.y))

    def get_cell(self, x, y):
        """Returns a cell from the cells list.

        @param x cell x coordinate
        @param y cell y coordinate
        @returns cell
        """
        return self.cells[x * self.grid_height + y]

    def get_adjacent_cells(self, cell):
        """Returns adjacent cells to a cell.

        Clockwise starting from the one on the right.

        @param cell get adjacent cells for this cell
        @returns adjacent cells list.
        """
        cells = []
        if cell.x < self.grid_width-1:
            cells.append(self.get_cell(cell.x+1, cell.y))
        if cell.y > 0:
            cells.append(self.get_cell(cell.x, cell.y-1))
        if cell.x > 0:
            cells.append(self.get_cell(cell.x-1, cell.y))
        if cell.y < self.grid_height-1:
            cells.append(self.get_cell(cell.x, cell.y+1))
        return cells

    def get_path(self):
        cell = self.end
        path = [(cell.x, cell.y)]
        while cell.parent is not self.start and cell.parent is not None:
            cell = cell.parent
            path.append((cell.x, cell.y))

        path.append((self.start.x, self.start.y))
        path.reverse()
        return path

    def update_cell(self, adj, cell):
        """Update adjacent cell.

        @param adj adjacent cell to current cell
        @param cell current cell being processed
        """
        adj.g = cell.g + 10
        adj.h = self.get_heuristic(adj)
        adj.parent = cell
        adj.f = adj.h + adj.g

    def solve(self):
        """Solve maze, find path to ending cell.

        @returns path or None if not found.
        """
        # add starting cell to open heap queue
        heapq.heappush(self.opened, (self.start.f, self.start))
        while len(self.opened):
            # pop cell from heap queue
            f, cell = heapq.heappop(self.opened)
            # add cell to closed list so we don't process it twice
            self.closed.add(cell)
            # if ending cell, return found path
            if cell is self.end:
                return self.get_path()
            # get adjacent cells for cell
            adj_cells = self.get_adjacent_cells(cell)
            for adj_cell in adj_cells:
                if adj_cell.reachable and adj_cell not in self.closed:
                    if (adj_cell.f, adj_cell) in self.opened:
                        # if adj cell in open list, check if current path is
                        # better than the one previously found
                        # for this adj cell.
                        if adj_cell.g > cell.g + 10:
                            self.update_cell(adj_cell, cell)
                    else:
                        self.update_cell(adj_cell, cell)
                        # add adj cell to open list
                        heapq.heappush(self.opened,(adj_cell.f, adj_cell))
class Astar_wrap(AStar):
    def init_grid(self,width,height,data,start,end,next_hour_data=None):
        """Prepare grid cells, walls.
        @param width grid's width.
        @param height grid's height.
        @param data wind speed data list of x,y tuples.
        @param start grid starting point x,y tuple.
        @param end grid ending point x,y tuple.
        @param next_hour_data wind speed data for next hour x,y tuples
        """
        self.grid_height = height
        self.grid_width = width
        if next_hour_data is None:
            next_hour_data = data
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if data[x][y] >= 15 :
                    reachable = False
                else:
                    target_dis = abs(end[0] - start[0]) + abs(end[1] - start[1])
                    pot_dis = abs(x - start[0]) + abs(y - start[1])
                    if target_dis > 30:
                        #why 30? because it can only run 30 steps every hour
                        if pot_dis == 30 and next_hour_data[x][y] >= 15:
                            reachable = False
                        else:
                            reachable = True
                    else:
                        if (pot_dis == target_dis and next_hour_data[x][y] >= 15) or (pot_dis == 30 and next_hour_data[x][y] >= 15):
                            reachable = False
                        else:
                            reachable = True
                self.cells.append(Cell(x, y, reachable))
        self.start = self.get_cell(*start)
        self.end = self.get_cell(*end) 
class Block():
    def __init__(self, x, y ,f):
        """Initialize new block.
        @param x cell x coordinate
        @param y cell y coordinate
        @param f distance to target
        """
        self.x = x
        self.y = y
        self.f = f
    # <
    def __lt__(self,other):
        return self.f < other.f
def handle_path_none(data,start,end,next_hour_data=None): 
    """When path is none between start and end,reset the target at middle point.
    @param data wind speed data list of x,y tuples.
    @param start grid starting point x,y tuple.
    @param end grid ending point x,y tuple.
    @param next_hour_data wind speed data for next hour."""
    temp_target = list(end[:2])
    path = None
    print("Handle path none at target:",end)
    if next_hour_data is None:
        next_hour_data = data
    point_list = []
    for x in range(-30+start[0],31+start[0]):
        for y in range(-30+start[1],31+start[1]):
            if x > 548-1 or y > 421-1 or x <0 or y<0:
                continue
            if (abs(x-start[0])+abs(y-start[1]))>30:
                continue
            if next_hour_data[x][y] <15 and data[x][y] <15:
                blk = Block(x,y,abs(x-end[0])+abs(y-end[1]))
                heapq.heappush(point_list,(blk.f, blk))           
    while len(point_list):
        f, blk = heapq.heappop(point_list)
        temp_target[0] = blk.x
        temp_target[1] = blk.y
        if start[0] == temp_target[0] and start[1] == temp_target[1]:#if start and target is the same,return start point
            print("start is the same as target!",start)
            return [start]
        print("Set target at:",temp_target)
        astar = Astar_wrap()
        astar.init_grid(548,421,data, start,temp_target,next_hour_data)
        path = astar.solve()
        if path is not None:
            return path
    return path#Dead Way!
def astar_path(myenv):
    """A start path finding
    @param myenv Env class to start the fram class
    @return path_record path record of the path dic."""
    paths = {item:[] for item in range(10)}
    path_record = {item:[] for item in range(10)}
    starts = []
    first = [True for x in range(10)]
    for j in range(10):
        starts.append(myenv.start_location)
    print("time:",myenv.time,"tick_per_hour",myenv.tick_per_hour)
    while not myenv.end():
        times_pass = myenv.time % myenv.tick_per_hour
        for i in range(10):
            if not myenv.done[i]:
                loc = myenv.locs[i]
                if first[i] and myenv.windspeed_at(loc) >= 15.0:
                    #do not take off
                    continue
                else:
                    first[i] = False
                #change map and get path every hour start
                if times_pass == 0:
                    print('path calculate in %d path at %d hour'%(i,myenv.get_curr_hour()))
                    astar = Astar_wrap() 
                    #set map, start, end
                    print("start:",starts[i],"target:",myenv.targets[i])
                    hour = myenv.get_curr_hour()
                    if hour == 17:
                        astar.init_grid(548, 421, myenv.datas[hour], starts[i], myenv.targets[i])
                    else:
                        astar.init_grid(548, 421, myenv.datas[hour], starts[i], myenv.targets[i],myenv.datas[hour+1]) 
                    #get A star path
                    paths[i]=astar.solve()
                    if paths[i] is None:
                        #it needs a function to handle none solution
                        hour = myenv.get_curr_hour()
                        print("None Solution in %d path at %d hour "%(i,hour))
                        if hour == 17:
                            paths[i] =handle_path_none(myenv.datas[hour],starts[i],myenv.targets[i])
                        else:
                            paths[i] =handle_path_none(myenv.datas[hour],starts[i],myenv.targets[i],myenv.datas[hour+1])
                        if paths[i] is None:
                            print("Dead Way!!!")
                            paths[i] = [starts[i]]
                        if len(paths[i]) == 1 and paths[i][0] == myenv.start_location:
                            #when the path is the same with the start,do not take off
                            paths[i] = [None for i in range(31)]
                            starts[i] = myenv.start_location
                            continue
                    #stand still 
                    while len(paths[i]) <= 30:
                        if paths[i][-1] is not None:
                            starts[i] = paths[i][-1]
                        paths[i].append(paths[i][-1])
                    if len(paths[i]) > 30 and paths[i][0] is not None:
                        #when we can not reach target in one hour ,we need reset the start position for the next hour
                        starts[i] = paths[i][30]
                        paths[i]=paths[i][:31]
                #change map and get path every hour end
                #change position into action_orders start
                if times_pass > len(paths[i])-1:
                    print("time_pass:",times_pass,"paths:",len(paths[i]))
                if paths[i][times_pass] is None:
                    myenv.set_move(i, None)
                else:
                    now_loc = paths[i][times_pass]
                    tar_loc = paths[i][times_pass+1]
                    x,y=(0,0)
                    if now_loc[0] > tar_loc[0]:
                        x = -1
                    elif now_loc[1] > tar_loc[1]:
                        y = -1
                    elif now_loc[0] < tar_loc[0]:
                        x = 1
                    elif now_loc[1] < tar_loc[1]:
                        y = 1
                    #change position into action_orders end
                    myenv.set_move(i, (x,y))
        myenv.tick()
        for i in range(10):
            path_record[i].append(myenv.locs[i])
    print('score %d time %d remain_task %d'%(myenv.score, myenv.time, myenv.remain_task))
    return path_record
def draw_path(myenv,path):
    """Draw the path.
    @param myenv Env class to start the fram class
    @param path path record of the path dic."""
    drawpath=path
    testenv =copy.deepcopy(myenv)
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
    x = []
    y = []
    plt.subplots(figsize=(25,20))
    for i in range(10):
        plt.scatter(x=[item[0] for item in drawpath[i] ],y=[item[1] for item in drawpath[i] ],c=color[i])
    plt.scatter(x=[item[0] for item in testenv.targets ],y=[item[1] for item in testenv.targets ],c='g')
    plt.grid(True,color='g',linestyle='--',linewidth='1')
    plt.savefig('path_Astar.png',dpi=100)
    plt.show()
def draw_map(myenv,hour):
    """Draw the map,red is unreachable.
    @param myenv Env class.
    @param hour hour of the map."""
    testenv = copy.deepcopy(myenv)
    paths = {item:[] for item in range(10)}
    astar = Astar_wrap() 
    astar.init_grid(548, 421, testenv.datas[hour], testenv.start_location, testenv.targets[0]) 
    x = []
    y = []
    for cell in astar.cells:
        if cell.reachable is False:
            x.append(cell.x)
            y.append(cell.y)
    plt.subplots(figsize=(100,80))
    plt.scatter(x,y,c='r')
    plt.scatter(x=[item[0] for item in testenv.targets ],y=[item[1] for item in testenv.targets ],c='g')
    plt.grid(True,color='g',linestyle='--',linewidth='1')
    plt.savefig('map%d.png'%(hour),dpi=100)
    plt.show()
if __name__ == '__main__':
    import copy
    myenv  = env.Env('tbl_TrueData4Test',6)
    testenv= copy.deepcopy(myenv)
    paths  = astar_path(testenv)
    testenv.dump()
    draw_path(myenv,paths)
    draw_map(myenv,1)
    
