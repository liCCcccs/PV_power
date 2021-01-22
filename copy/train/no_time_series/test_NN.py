import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Train_all = True
Feature_name = "time"
Feature_dict = {"date": 0, "time": 1, "wind_speed": 2, "temperature": 3, "radiation": 4, "wind_direction": 5, "rainfall": 6,
                         "max_wind_speed": 7, "air_pressure": 8, "hail_accumulation": 9, "pyranometer_1": 10,
                         "temperature_probe_1": 11, "temperature_probe_2": 12, "AEDR": 13, "Active_Power": 14}

def main():
    valid_dir = "../../Yulara/read_to_train/MinMaxScaler/"
    valid_filename = "Yulara_2020.csv"
    data = np.loadtxt(valid_dir + valid_filename, delimiter=",", dtype=float)[0:290, :]
    if Train_all:
        X_test, y_test = data[:, :-1], data[:, -1]
    else:
        feature_id = Feature_dict[Feature_name]
        X_test, y_test = data[:, feature_id], data[:, -1]

    if Train_all:
        model = keras.models.load_model("./saved_model/NN_all.h5")
    else:
        model = keras.models.load_model("./saved_model/NN_" + Feature_name + ".h5")
    y_pred = model.predict(X_test)

    plt.plot(y_pred)
    plt.plot(y_test / 1000)
    plt.show()




if __name__ == '__main__':
    main()