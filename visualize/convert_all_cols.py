import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


TIME_COL = 1  # column number representing time
ENERGY_COL = 2  # energy
EFF_COL = 3  # efficiency
POWER_COL = 4  # column number representing power
AVE_COL = 5  # average
NORM_COL = 6  # normalized
TEMP_COL = 7  # column number representing power
VOLT_COL = 8  # voltage
EU_COL = 9  # energy used
PU_COL = 10  # power used

NUM_FEATURE = 10


def process_time(time):
    """ Reformat time to 24 hour format, convert minutes to hour decimals """
    return float(time[0]) + round(float(time[1][:-2]) / 60.0, 2)

def fill_missing(col_name):
    pass


def read_csv_to_numpy(filename, prefix):
    raw_data = pd.read_csv(prefix + filename, sep=',', header=0).values  # read to a list
    data_length = raw_data.shape[0]
    my_data = np.zeros([data_length, NUM_FEATURE])   # create a empty numpy array for processed data

    # Process time
    for line in range(data_length):
        raw_time = raw_data[line, TIME_COL].split(":")
        # std_time = process_time(raw_time) if raw_time[1][-2:] == "AM" else process_time(raw_time) + 12.0
        if raw_time[1][-2:] == "AM" and raw_time[0] != "12":
            std_time = process_time(raw_time)
        elif raw_time[1][-2:] == "AM" and raw_time[0] == "12":
            std_time = process_time(raw_time) - 12.0
        elif raw_time[1][-2:] == "PM" and raw_time[0] != "12":
            std_time = process_time(raw_time) + 12.0
        else:  # 12 PM
            std_time = process_time(raw_time)

        # Process power
        if raw_data[line, POWER_COL] == "-":
            print(line)  # just in case there are some missing power data
        std_power = float(raw_data[line, POWER_COL][:-1].replace(",", ""))

        # Process average
        if raw_data[line, AVE_COL] is not "-":
            std_ave = float(raw_data[line, AVE_COL][:-1].replace(",", ""))
        elif line == 0:
            i = 0
            while raw_data[line + i, AVE_COL] == "-":
                i += 1
            std_ave = float(raw_data[line + i, AVE_COL][:-1].replace(",", ""))
        else:
            std_ave = my_data[line - 1, AVE_COL-1]

        # Process normalized
        if raw_data[line, NORM_COL] is not "-":
            std_norm = float(raw_data[line, NORM_COL][:-5])
        elif line == 0:
            i = 0
            while raw_data[line + i, NORM_COL] == "-":
                i += 1
            std_norm = float(raw_data[line + i, NORM_COL][:-5])
        else:
            std_norm = my_data[line - 1, NORM_COL-1]

        # Process temperature
        if raw_data[line, TEMP_COL] is not "-":
            std_temperature = float(raw_data[line, TEMP_COL][:-1])
        elif line == 0:
            i = 0
            while raw_data[line + i, TEMP_COL] == "-":
                i += 1
            std_temperature = float(raw_data[line + i, TEMP_COL][:-1])
        else:
            std_temperature = my_data[line - 1, TEMP_COL-1]

        # TIME_COL is the 0th column, so minus one for all columns
        my_data[data_length - line - 1, TIME_COL-1] = std_time   # 1. time
        my_data[data_length - line - 1, ENERGY_COL-1] = raw_data[line, ENERGY_COL][:-3]  # 2. energy (remove kWh)
        my_data[data_length - line - 1, EFF_COL-1] = raw_data[line, EFF_COL][:-6]  # 3. efficiency (remove kWh/kW)
        my_data[data_length - line - 1, POWER_COL-1] = std_power   # 4. power
        my_data[data_length - line - 1, AVE_COL-1] = std_ave  # 5. average
        my_data[data_length - line - 1, NORM_COL-1] = std_norm  # 6. normalized
        my_data[data_length - line - 1, TEMP_COL-1] = std_temperature  # 7. temperature
        my_data[data_length - line - 1, VOLT_COL-1] = raw_data[line, VOLT_COL][:-1]  # 8. voltage (remove V)
        my_data[data_length - line - 1, EU_COL-1] = raw_data[line, EU_COL][:-3]  # 9. eu (remove kWh)
        my_data[data_length - line - 1, PU_COL-1] = raw_data[line, PU_COL][:-1].replace(",", "")  # 10. pu (remove W)

    np.savetxt("../data/processed/data_full/" + filename, my_data, delimiter=",")


def main():
    raw_data_dir = "../data/raw_data/"
    all_files = [f for f in listdir(raw_data_dir) if isfile(join(raw_data_dir, f))]

    for filename in sorted(all_files):
        print("Processing: " + filename)
        read_csv_to_numpy(filename, prefix=raw_data_dir)


if __name__ == '__main__':
    main()
