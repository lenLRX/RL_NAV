#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <queue>
#include <cstring>
#include <cstdlib>

const int dim_x = 548;
const int dim_y = 421;

const int start_x = 141;
const int start_y = 327;

const int total_days = 5;
const int total_hours = 18;



using namespace std;

class node
{
public:
    node(int x = -1, int y = -1, float wind_speed = -1) :
        x(x), y(y), wind_speed(wind_speed), time(-1), from_x(-1), from_y(-1), visited(false)
    {
    }

    void set_from(node* from_node)
    {
        from_x = from_node->x;
        from_y = from_node->y;
        visited = true;
    }

    int x;
    int y;
    float wind_speed;
    int time;

    int from_x;
    int from_y;
    int from_time;
    bool visited;

};

class Solver
{
public:
    Solver()
    {
        csv_file = ofstream("D:\\result.csv");
        datas.resize(total_days);
        for (int day = 0; day < total_days; ++day)
        {
            auto& data_of_day = datas[day];
            data_of_day.resize(total_hours);
            for (int hour = 0; hour < total_hours; ++hour)
            {
                auto& data_of_hour = data_of_day[hour];
                data_of_hour.resize(dim_x);
                for (int x = 0; x < dim_x; ++x)
                {
                    auto& row = data_of_hour[x];
                    row.resize(dim_y);
                }
            }
        }
    }

    ~Solver()
    {
        csv_file.close();
    }

    void load_data(char* path)
    {
        ifstream csv_file(path);
        char line_buffer[256];
        int line_count = 0;
        //skip header
        csv_file.getline(line_buffer, 256);
        while (csv_file.getline(line_buffer,256))
        {
            int col = 0;
            string token;
            stringstream ss(line_buffer);
            int x, y, day, hour;
            float wind_speed;
            while (std::getline(ss, token, ','))
            {
                switch (col)
                {
                case 0:
                    x = stoi(token) - 1;
                    break;
                case 1:
                    y = stoi(token) - 1;
                    break;
                case 2:
                    day = stoi(token) - 6;
                    break;
                case 3:
                    hour = stoi(token) - 3;
                    break;
                case 4:
                    wind_speed = stof(token);
                    break;
                default:
                    break;
                }
                ++col;
            }
            line_count++;
            datas[day][hour][x][y] = node(x,y,wind_speed);
            if (0 == line_count % 10000)
                cout << line_count / (548.0 * 421.0 * 18 * 5) << endl;
        }
        cout << "load data done " << line_count << " loaded" << endl;
    }

    void output_csv(int target_no, int day, int t, int target_x, int target_y)
    {
        node* n = &ext_datas[t][target_x][target_y];
        vector<string> lines;
        while (true)
        {
            char buffer[1024] = {0};
            //targetno, day, time, x,y
            sprintf(buffer, "%d,%d,%02d:%02d,%d,%d\n", target_no + 1, day + 6, t / 30 + 3, (t % 30) * 2, n->x + 1, n->y + 1);
            lines.push_back(buffer);
            if (n->x == start_x && n->y == start_y)
            {
                break;
            }
            if (n->from_x < 0 || n->from_y < 0)
            {
                cerr << "error idx" << endl;
            }
            t--;
            n = &ext_datas[t][n->from_x][n->from_y];
        }

        for (int i = lines.size() - 1; i >= 0; --i)
        {
            csv_file << lines[i];
        }
    }

    void init(int day)
    {
        ext_datas.clear();
        for (int hour = 0; hour < total_hours; ++hour)
        {
            for (int i = 0; i < 30; ++i)
            {
                ext_datas.push_back(datas[day][hour]);
            }
        }
    }

    void try_node(node* from_node, node* n)
    {
        if (n->wind_speed >= max_spd)
        {
            return;
        }

        if (true == n->visited)
        {
            return;
        }

        n->set_from(from_node);
        Q_next.push(n);
    }

    void try_nearby_node(node* n, int t)
    {
        if (n->x - 1 >= 0)
        {
            try_node(n, &ext_datas[t][n->x - 1][n->y]);
        }

        if (n->x + 1 < dim_x)
        {
            try_node(n, &ext_datas[t][n->x + 1][n->y]);
        }

        if (n->y - 1 >= 0)
        {
            try_node(n, &ext_datas[t][n->x][n->y - 1]);
        }

        if (n->y + 1 < dim_y)
        {
            try_node(n, &ext_datas[t][n->x][n->y + 1]);
        }

        //stay here
        try_node(n, &ext_datas[t][n->x][n->y]);
    }

    void solve(vector<pair<int, int>> targets)
    {
        max_spd = 0.7f;
        int score = 0;
        int success = 0;
        for (int i = 0; i < total_days; ++i)
        {
            
            for (int target_no = 0; target_no < 10; ++target_no)
            {
                auto p = targets[target_no];
                int t = solve(i, p.first, p.second);
                if (t >= 0)
                {
                    output_csv(target_no, i, t, p.first, p.second);
                    score += t * 2;
                    success++;
                }
                else
                {
                    score += 24 * 60;
                    cerr << "crashed" << endl;
                }
            }
        }
        cout << "score " << score << " success: " << success << endl;
        system("pause");
    }

    int solve(int day, int target_x, int target_y)
    {
        
        for (int start_hour = 0; start_hour < total_hours; ++start_hour)
        {
            init(day);
            Q = queue<node*>();
            Q_next = queue<node*>();
            int start_t = start_hour * 30;
            node* start_node = &ext_datas[start_t][start_x][start_y];
            start_node->time = start_t;
            start_node->visited = true;
            Q_next.push(start_node);
            for (int t = start_hour * 30; t < total_hours * 30; ++t)
            {
                Q = Q_next;
                Q_next = queue<node*>();
                if (Q.empty())
                {
                    break;
                }
                while (!Q.empty())
                {
                    node* n = Q.front();
                    Q.pop();
                    if (t + 1 < total_hours * 30)
                    {
                        try_nearby_node(n, t + 1);
                    }
                }

                if (t + 1 < total_hours * 30 && ext_datas[t + 1][target_x][target_y].visited)
                {
                    cout << "(" << target_x << "," << target_y << ") done at t=" << t + 1 << endl;
                    return t + 1;
                }
                
            }
        }
        return -1;
    }

    float max_spd;

    queue<node*> Q;
    queue<node*> Q_next;

    vector<vector<vector<vector<node>>>> datas;
    vector<vector<vector<node>>> ext_datas;

    ofstream csv_file;
};


int main(int argc, char** argv)
{
    if (2 != argc)
    {
        cerr << "usage: ./3DBFS path_to_wind.csv" << endl;
        return -1;
    }
    Solver solver;
    solver.load_data(argv[1]);
    solver.solve({
        {83,202},
        {198,370},
        {139,233},
        {235,240},
        {314,280},
        {357,206},
        {362,236},
        {422,265},
        {124,374},
        {188,273}
        });
    return 0;
}