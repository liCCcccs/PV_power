import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import csv


Train_filepaths = ["./data/PV_12_11.csv"]
TIME_COL = 1  # column number representing time
Energy_COL = 2
Eff_COL = 3
POWER_COL = 4  # column number representing power
Ave_COL = 5
Norm_COL = 6
TEMP_COL = 7  # column number representing power
VOLT_COL = 8
EU_COL = 9
PU_COL = 10


def process_time(time, offset=0):
    """ Reformat time to 24 hour format, convert minutes to hour decimals """
    return str(int(time[0]) + offset) + ":" + time[1][:-2]

def process_cell(data_length, raw_data, my_data, line, col, suffix_n):
    # print(line)
    if raw_data[line, col] is not "-":
        std_data = raw_data[line, col][:-suffix_n]
    elif line == 0:
        i = 0
        while raw_data[line + i, col] == "-":
            i += 1
        std_data = raw_data[line + i, col][:-suffix_n]
    else:
        std_data = my_data[data_length - line, col-1]

    try:
        std_data = std_data.decode('ascii')
    except:
        std_data = str(std_data)
    return float(std_data.replace(",", ""))


def read_csv_to_numpy(filename, prefix):
    raw_data = pd.read_csv(prefix + filename, sep=',', header=0).values  # read to a list
    data_length = raw_data.shape[0]
    my_data = np.zeros([data_length, 10], dtype=object)   # create a empty numpy array for processed data
    for line in range(data_length):
        raw_time = raw_data[line, TIME_COL].split(":")
        # std_time = process_time(raw_time) if raw_time[1][-2:] == "AM" else process_time(raw_time) + 12.0
        if raw_time[1][-2:] == "AM" and raw_time[0] != "12":
            std_time = process_time(raw_time)
        elif raw_time[1][-2:] == "AM" and raw_time[0] == "12":
            std_time = process_time(raw_time, offset=-12)
        elif raw_time[1][-2:] == "PM" and raw_time[0] != "12":
            std_time = process_time(raw_time, offset=12)
        else:  # 12 PM
            std_time = process_time(raw_time)

        raw_date = raw_data[line, 0][2:].split("/")
        raw_date.reverse()
        # print(raw_date)
        raw_date_str = "/".join(raw_date)
        std_date_time = "20" + raw_date_str + " " + std_time

        Energy_COL = 2
        Eff_COL = 3
        POWER_COL = 4  # column number representing power
        Ave_COL = 5
        Norm_COL = 6
        TEMP_COL = 7  # column number representing power
        VOLT_COL = 8
        EU_COL = 9
        PU_COL = 10

        std_1 = process_cell(data_length, raw_data, my_data, line, Energy_COL, suffix_n=3)
        std_2 = process_cell(data_length, raw_data, my_data, line, Eff_COL, suffix_n=6)
        std_3 = process_cell(data_length, raw_data, my_data, line, POWER_COL, suffix_n=1)
        std_4 = process_cell(data_length, raw_data, my_data, line, Ave_COL, suffix_n=1)
        std_5 = process_cell(data_length, raw_data, my_data, line, Norm_COL, suffix_n=5)
        std_6 = process_cell(data_length, raw_data, my_data, line, TEMP_COL, suffix_n=1)
        std_7 = process_cell(data_length, raw_data, my_data, line, VOLT_COL, suffix_n=1)
        std_8 = process_cell(data_length, raw_data, my_data, line, EU_COL, suffix_n=3)
        std_9 = process_cell(data_length, raw_data, my_data, line, PU_COL, suffix_n=1)


        my_data[data_length - line - 1, 0] = str(std_date_time)
        my_data[data_length - line - 1, 1] = str(std_1)
        my_data[data_length - line - 1, 2] = str(std_2)
        my_data[data_length - line - 1, 3] = str(std_3)
        my_data[data_length - line - 1, 4] = str(std_4)
        my_data[data_length - line - 1, 5] = str(std_5)
        my_data[data_length - line - 1, 6] = str(std_6)
        my_data[data_length - line - 1, 7] = str(std_7)
        my_data[data_length - line - 1, 8] = str(std_8)
        my_data[data_length - line - 1, 9] = str(std_9)


    # np.savetxt("./data/processed/test_data/" + filename, my_data, delimiter=",", fmt="%s")
    with open("./data/processed/test_data/" + filename, 'w') as f:
        csv.writer(f).writerows(my_data)


def main():
    raw_data_dir = "./data/test_data/"
    all_files = [f for f in listdir(raw_data_dir) if isfile(join(raw_data_dir, f))]

    for filename in all_files:
        print("Processing: " + filename)
        read_csv_to_numpy(filename, prefix=raw_data_dir)


if __name__ == '__main__':
    main()
