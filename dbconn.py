import MySQLdb

import json
import os
import csv

def get_conn(auth_path = 'auth.json'):
    with open(auth_path) as fp:
        auth = json.load(fp)
        return MySQLdb.connect(**auth)

def query_true_data(conn, day, hour):
    c = conn.cursor()
    try:
        c.execute('use %s;'%'data')
        c.execute('select * from tbl_TrueData where day=%d and hour=%d'%(day,hour))
        datas = c.fetchall()
        print(type(datas),len(datas))
        print(datas[0])
    finally:
        c.close()

def query_data_by_table(conn, table, day, hour, modelno = None):
    c = conn.cursor()
    try:
        c.execute('use %s;'%'data')
        SQLstmt = 'select * from %s where day=%d and hour=%d'%(table,day,hour)
        if not modelno is None:
            SQLstmt = SQLstmt + 'and %d'%modelno
        print(SQLstmt)
        c.execute(SQLstmt)
        return c.fetchall()
    finally:
        c.close()


def drop_init_db(conn, dbname = 'data'):
    c = conn.cursor()
    print('drop db: %s'%dbname)
    c.execute('DROP DATABASE IF EXISTS %s;'%(dbname,))
    print('create db: %s'%dbname)
    c.execute('CREATE DATABASE %s;'%(dbname,))
    c.close()

def drop_create_tbl_ForecastData(conn, dbname = 'data'):
    try:
        c = conn.cursor()
        conn.autocommit(False)
        c.execute('use %s;'%dbname)
        tbl_name = 'tbl_ForecastData'
        print('drop tbl: %s'%tbl_name)
        c.execute('DROP TABLE IF EXISTS %s;'%(tbl_name,))
        print('create tbl: %s'%tbl_name)
        c.execute('''CREATE TABLE IF NOT EXISTS %s (
        x SMALLINT,
        y SMALLINT,
        day SMALLINT,
        hour TINYINT,
        modelno TINYINT,
        windspeed FLOAT);'''%(tbl_name,))

        with open(os.path.join('data', 'ForecastDataforTesting_20171205',
            'ForecastDataforTesting_201712.csv'), newline='') as csvfile:
            testreader = csv.reader(csvfile)
            row_num = 0
            for row in testreader:
                row_num = row_num + 1
                if row_num == 1:
                    continue
                print(row_num,len(testreader))
                c.execute('''INSERT INTO tbl_ForecastData (x, y, day, hour, modelno, windspeed)
                VALUES (%s, %s, %s, %s, %s, %s);'''%tuple(row))
    except Exception as e:
        print(e)
    finally:
        c.close()
        conn.commit()
        conn.close ()

if __name__ == '__main__':
    conn = get_conn()
    query_true_data(conn,5,10)