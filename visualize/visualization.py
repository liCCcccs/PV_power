"""
Visualize data, save to figure
"""
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def concat_data_scale(raw_dir, all_files):
    """  Concatenate multiple files into one file, and fit a scaler """
    isFirst = 1
    for file in sorted(all_files):
        if isFirst == 1:
            concat_data = np.array(pd.read_csv(raw_dir + file, sep=',', header=None).values)  # read to a np arrau
        else:
            another_data = np.array(pd.read_csv(raw_dir + file, sep=',', header=None).values)  # read to a np arrau
            concat_data = np.append(concat_data, another_data, axis=0)
        isFirst = 0

    scaler = MinMaxScaler()
    scaler.fit(concat_data)
    return scaler


def visualize_data(scaler, raw_dir, all_files):
    """ scale and plot and save"""
    TIME_COL = 0  # column for time
    for file in sorted(all_files):
        raw_data = np.array(pd.read_csv(raw_dir + file, sep=',', header=None).values)  # read to a list
        scaled_data = scaler.transform(raw_data)
        plt.figure(figsize=(12, 7))
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 1], color='black', linestyle='-', label="Energy")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 2], color='blue', linestyle='-', label="Efficiency")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 6], color='red', linestyle='-', label="Temperature")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 7], color='black', linestyle='--', label="Voltage")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 8], color='blue', linestyle='--', label="EU")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 9], color='red', linestyle='--', label="PU")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 4], color='black', linestyle=':', label="Average")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 5], color='blue', linestyle=':', label="Normalized")
        plt.plot(raw_data[:, TIME_COL], scaled_data[:, 3], color='green', linestyle=':', label="Power")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Scaled value")
        # plt.show()
        plt.savefig("./saved_fig/" + file[:-4] + ".png", bbox_inches="tight")  # [:-4] is to remove the ".csv", not an emoji


def main():
    raw_dir = "../data/processed/data_full/"
    all_files = [f for f in listdir(raw_dir) if isfile(join(raw_dir, f))]

    my_scaler = concat_data_scale(raw_dir, all_files)
    visualize_data(my_scaler, raw_dir, all_files)


if __name__ == '__main__':
    main()
