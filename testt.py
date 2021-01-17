import numpy as np
import csv

start_date = "2016/4/7"
end_date = "2016/4/10"

file = open("./data/new_data.csv")
ttime = np.loadtxt("./data/new_data.csv", delimiter=",", dtype=str)


print(ttime.shape)
file.close()
