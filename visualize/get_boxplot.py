"""
Visualize data, save to figure
"""
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


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
    return scaler, concat_data


def visualize_data(scaler, concat_data, plot_option=1):
    """ scale and plot and save"""
    scaled_data = scaler.transform(concat_data)
    extracted_data = np.c_[np.around(concat_data[:, 0], decimals=2), scaled_data[:, 3]]   # extract time:0 , power:3

    if plot_option == 1:
        # Plot 0-12 hours
        row_tobe_del = []
        for row in range(extracted_data.shape[0]):
            if extracted_data[row, 0] > 12.1:
                row_tobe_del.append(row)
        box_data = np.delete(extracted_data, row_tobe_del, axis=0)
        print(len(row_tobe_del))
        print(box_data.shape)

        plt.figure(figsize=(15, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.boxplot(x=box_data[:, 0], y=box_data[:, 1], manage_ticks=False)
        ax.set_xticks(np.arange(0, 13)*12 - 1)
        plt.tight_layout()
        # plt.show()
        plt.savefig("./saved_fig/box_1st_half.png")

    elif plot_option == 2:
        # Plot 12-24 hours
        row_tobe_del = []
        for row in range(extracted_data.shape[0]):
            if extracted_data[row, 0] < 11.9:
                row_tobe_del.append(row)
        box_data = np.delete(extracted_data, row_tobe_del, axis=0)
        print(len(row_tobe_del))
        print(box_data.shape)

        plt.figure(figsize=(15, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.boxplot(x=box_data[:, 0], y=box_data[:, 1], manage_ticks=False)
        ax.set_xticks(np.arange(0, 13)*12 + 1)
        plt.tight_layout()
        # plt.show()
        plt.savefig("./saved_fig/box_2nd_half.png")

    elif plot_option == 3:
        # Plot 5.5 - 18.5 hours
        row_tobe_del = []
        for row in range(extracted_data.shape[0]):
            if extracted_data[row, 0] < 5.4 or extracted_data[row, 0] > 20.7:
                row_tobe_del.append(row)
        box_data = np.delete(extracted_data, row_tobe_del, axis=0)
        print(len(row_tobe_del))
        print(box_data.shape)

        plt.figure(figsize=(15, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.boxplot(x=box_data[:, 0], y=box_data[:, 1], manage_ticks=False)
        ax.set_xticks(np.arange(1, 17)*12 - 5)
        plt.tight_layout()
        # plt.show()
        plt.savefig("./saved_fig/box_middle.png")


def main():
    raw_dir = "../data/processed/data_full/"
    all_files = [f for f in listdir(raw_dir) if isfile(join(raw_dir, f))]

    my_scaler, concat_data = concat_data_scale(raw_dir, all_files)
    visualize_data(my_scaler, concat_data, plot_option=3)


if __name__ == '__main__':
    main()
