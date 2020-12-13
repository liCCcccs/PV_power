import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


def concat_data():
    """  Concatenate multiple files into one file """
    raw_dir = "../data/processed/data_2in1out/"
    all_files = [f for f in listdir(raw_dir) if isfile(join(raw_dir, f))]

    isFirst = 1
    for file in sorted(all_files):
        if isFirst == 1:
            concat_data = np.array(pd.read_csv(raw_dir + file, sep=',', header=None).values)  # read to a list
        else:
            another_data = np.array(pd.read_csv(raw_dir + file, sep=',', header=None).values)  # read to a list
            concat_data = np.append(concat_data, another_data, axis=0)
        isFirst = 0

    concat_name = "raw_data.csv"
    np.savetxt("../data/data_seq2seq/data_2features/" + concat_name, concat_data, delimiter=",")


def diff_data():
    """ Convert data to: X: yesterday's power, today's time, today's temperature; y: today's power """
    pass  # may be deprecated


def main():
    concat_data()
    diff_data()


if __name__ == '__main__':
    main()
