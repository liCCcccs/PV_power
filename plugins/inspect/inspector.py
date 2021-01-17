import logging
import numpy as np
from numpy import genfromtxt
from itertools import chain
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Inspector():
    def __init__(self, filepath, option, start_date, end_date):
        self._filepath = filepath
        self._option = option
        self._start_date = start_date  # format "2020/1/24"
        self._end_date = end_date
        self._read_from_csv()
        print("=======", self._data.shape)
        self._feature = {"wind_speed": 0, "temperature": 1, "radiation": 2, "wind_direction": 3, "rainfall": 4,
                         "max_wind_speed": 5, "air_pressure": 6, "hail_accumulation": 7, "pyranometer_1": 8,
                         "temperature_probe_1": 9, "temperature_probe_2": 10, "AEDR": 11, "Active_Power": 12}
        self._inv_feature = {v: k for k, v in self._feature.items()}

    def _read_from_csv(self):
        try:
            print("start reading")
            self._data = np.loadtxt(self._filepath, delimiter=",", dtype=float)
            self._time = np.loadtxt("./data/new_date.csv", delimiter=",", dtype=str)
            print("=======****", self._data.shape)
        except IOError:
            raise Exception("File %s not found", self._filepath)  # TODO: format print

    def process(self):
        print("===========", self._start_date)
        if self._start_date is None and self._end_date is None:
            pass  # use all data
        elif self._end_date is None:
            # single day
            self._end_date = self._start_date
            self._data = self._filter_data()
        else:
            self._data = self._filter_data()
        print("=======", self._data.shape)

        if self._option == 1:
            logger.info("plot power and factor")
            # TODO: implement plot in a another file
            self._plot_2(self._feature["Active_Power"], self._feature["radiation"])
        if self._option == 2:
            n = self._data.shape[0]
            print("Num of points = ", n)
            self._get_relevance(n)

    def _filter_data(self):
        # TODO: if the input start date or end date not exist?
        start_row, end_row = -1, -1
        for index, row in enumerate(self._time):
            if self._start_date in row:
                if start_row == -1:
                    start_row = index
            elif self._end_date in row:
                end_row = index
            elif end_row != -1:
                break
        print("=============", start_row, end_row)

        filtered = self._data[start_row:end_row+1, :]
        logger.info("%d data points been used", filtered.shape[0])
        logger.info("from %s, to %s", self._data[start_row, 0], self._data[end_row, 0])
        return filtered

    def _plot_2(self, col1, col2):
        plt.plot(self._data[:, col1])
        plt.plot(self._data[:, col2])
        plt.show()

    def _get_relevance(self, n):
        x = self._data[:, self._feature["Active_Power"]]
        for i in range(11):
            y = self._data[:, i]
            r = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / np.sqrt((n * np.sum(x*x) - np.sum(x)*(np.sum(x))) * (n * np.sum(y*y) - np.sum(y)*(np.sum(y))) )
            print("Relevance: power and " + self._inv_feature[i], r)
