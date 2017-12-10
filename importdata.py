import csv
import os

def test_reader():
    with open(os.path.join('data', 'ForecastDataforTesting_20171205',
        'ForecastDataforTesting_201712.csv'), newline='') as csvfile:
        testreader = csv.reader(csvfile)
        for row in testreader:
            print(row)

if __name__ == "__main__":
    test_reader()