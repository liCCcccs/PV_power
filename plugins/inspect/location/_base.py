import logging
import os
import numpy as np
import re
import datetime
import csv
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LocationBase():
    def __init__(self, filepath, option, start_date, end_date):
        # TODO: is_all... maybe we don't
        self._filepath = filepath
        self._option = option
        self._start_date = start_date
        self._end_date = end_date
        self._feature = None   # implemented in children class
        self._inv_feature = None   # implemented in children class
        self._date_separator = None  # implemented in children class
        print("========", self._date_separator)

    def _write_start_end_date(self):
        start_args = (int(x) for x in re.split(self._date_separator, self._start_date))
        end_args = (int(x) for x in re.split(self._date_separator, self._end_date))
        self._start_date = datetime.datetime(*start_args)
        self._end_date = datetime.datetime(*end_args)

    def _file_to_read(self):
        years = list(range(self._start_date.year, self._end_date.year + 1))
        data_name = os.path.split(self._filepath)[-1]
        files = [os.path.join(self._filepath, data_name + "_" + str(year) + ".csv") for year in years]
        self._files = files

    def _load_data_from_file(self):
        if len(self._files) == 0:
            raise ValueError("Please input correct year.")
        elif len(self._files) == 1:
            try:
                start_row, end_row = self._search_two_rows(self._files[0])
                self._data = self._read_filter_csv(0, start_row, end_row)
            except IOError:
                raise Exception("File not found: ", self._files[0])
        else:
            for index, file in enumerate(self._files):
                try:
                    if index == 0:
                        start_row = self._search_one_row(self._files[index], is_start=True)
                        self._data = self._read_filter_csv(index, start_row=start_row)
                    elif index == len(self._files) - 1:
                        end_row = self._search_one_row(self._files[index], is_start=False)
                        data = self._read_filter_csv(index, end_row=end_row)
                        self._data = np.vstack((self._data, data))
                    else:
                        data = self._read_filter_csv(index)
                        self._data = np.vstack((self._data, data))
                except IOError:
                    raise Exception("File not found: ", self._files[index])

    def process(self):
        self._write_start_end_date()
        self._file_to_read()
        self._load_data_from_file()
        print("Number of data points been inspected: ", self._data.shape[0])

        if self._option == 1:
            logger.info("plot power and factor")
            # TODO: implement plot in a another file
            self._plot_2(self._feature["Active_Power"], self._feature["radiation"])
        if self._option == 2:
            n = self._data.shape[0]
            print("Num of points = ", n)
            self._get_relevance(n)

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

    def _read_filter_csv(self, index, start_row=0, end_row=None):
        # TODO: return the actual start and end date
        data = np.loadtxt(self._files[index], delimiter=",", dtype=object)[:, 1:].astype(float)
        if end_row is None:
            data = data[start_row:, :]
        else:
            data = data[start_row: end_row + 1, :]
        return data

    def _search_one_row(self, filename, is_start):
        """ Return the row number of the start_date or end_date in the file
            If the target date doesn't exist, return the closest later date after start date
        """
        target_date = self._start_date if is_start else self._end_date
        target_row = -1
        with open(filename) as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                curr_date_args = (int(x) for x in re.split("[/ :]", row[0]))
                curr_date = datetime.datetime(*curr_date_args)
                if target_row == -1 and curr_date >= target_date:
                    target_row = index
                    break
        return target_row

    def _search_two_rows(self, filename):
        """ Return the number of rows of start and end date in the file """
        start_row, end_row = -1, -1
        with open(filename) as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                curr_date_args = (int(x) for x in re.split("[/ :]", row[0]))
                curr_date = datetime.datetime(*curr_date_args)
                if start_row == -1 and curr_date >= self._start_date:
                    start_row = index
                if end_row == -1 and curr_date >= self._end_date:
                    end_row = index
                    break
        return start_row, end_row
