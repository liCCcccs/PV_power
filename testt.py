import numpy as np
import csv
import time
import datetime
import re
import logging

Start_date = "2016/4/7 0:00"
End_date = "2016/5/7 1:55"
# TODO: check end_date larger than start_date

# start_time = time.time()
file = open("./data/Yulara/Yulara_2016.csv")
# ttime = np.loadtxt("./data/new_data.csv", delimiter=",", dtype=str)
# print("--- %s seconds ---" % (time.time() - start_time))

logger = logging.getLogger(__name__)

start_date_args = (int(x) for x in re.split("[/ :]", Start_date))
end_date_args = (int(x) for x in re.split("[/ :]", End_date))
start_date = datetime.datetime(*start_date_args)
end_date = datetime.datetime(*end_date_args)
filtered_data = None

start_row = -1
end_row = -1
reader = csv.reader(file)

start_time = time.time()
for index, row in enumerate(reader):
    curr_date_args = (int(x) for x in re.split("[/ :]", row[0]))
    curr_date = datetime.datetime(*curr_date_args)
    if start_row == -1 and curr_date >= start_date:
        start_row = index
        # filtered_data = np.array([float(x) for x in row[1:]])
        continue
    # if start_row != -1 and end_row == -1:
        # filtered_data = np.vstack((filtered_data, [float(x) for x in row[1:]]))
    if end_row == -1 and curr_date >= end_date:
        end_row = index
        break

print("--- %s seconds ---" % (time.time() - start_time))

file.close()

print(start_row)
print(end_row)
# print(filtered_data.shape)


start_time = time.time()
my_data = np.loadtxt("./data/Yulara/Yulara_2016.csv", delimiter=",", dtype=object)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
my_data = np.loadtxt("./data/Yulara/Yulara_2016.csv", delimiter=",", dtype=object)[:, 1:].astype(float)
print("--- %s seconds ---" % (time.time() - start_time))
print(my_data[0, 1:])

