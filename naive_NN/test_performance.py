import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def main():
    valid_dir = "../data/data_2in1out/valid/"
    valid_filename = "PV_11_15.csv"
    raw_data = pd.read_csv(valid_dir + valid_filename, sep=',', header=None).values  # read to a list
    print(raw_data[0:3, :])

    model = keras.models.load_model("./saved_model/NN_best.h5")
    y_pred = model.predict(raw_data[:, :2])

    plt.plot(y_pred)
    plt.plot(raw_data[:, 2] / 10000)
    plt.show()




if __name__ == '__main__':
    main()
