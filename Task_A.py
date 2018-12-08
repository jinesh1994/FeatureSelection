import numpy as np
import pandas as pd


def load_data(file_path):
    data_frame = pd.read_csv(file_path, header=None, sep=",")
    class_numbers = data_frame.iloc[:1, :]
    return data_frame


if __name__ == '__main__':
    load_data('lris.txt')
