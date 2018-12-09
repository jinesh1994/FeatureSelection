import numpy as np
import pandas as pd


def load_data(file_path):
    try:
        if 'Genome' in file_path:
            data_frame = pd.read_csv(file_path, header=None, sep=",")
            class_numbers = data_frame.iloc[:1, :]
            only_data_frame = data_frame.iloc[1:, :]
            print('Data has been cleaned')
            return data_frame, class_numbers, only_data_frame
        elif 'lris' in file_path:
            data_frame = pd.read_csv(file_path, header=None, sep=",")
            class_numbers = data_frame.iloc[:, -1]
            only_data_frame = data_frame.iloc[:, :-1]
            only_data_frame.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
            print('Data has been cleaned')
            return data_frame, class_numbers, only_data_frame
    except Exception as e:
        print(e)


def f_test_for_iris(data, label):
    try:
        # column wise mean of the dataframe.
        data_mean = data.mean()
        data_frame_size = int(data.size)
        data_mean_float = []
        for i in data_mean:
            data_mean_float.append(float(i))
        explained_variance = 0
        k = len(data.columns.values)
        overall_data_mean = np.mean(data_mean_float)
        for i in range(len(data.columns.values)):
            n = data[data.columns.values[i]].shape[0] - 100
            explained_variance = explained_variance + ((n * (data_mean_float[i] - float(overall_data_mean)) ** 2) / k - 1)
        print('explained_variance is: ', explained_variance)
        container = {}
        # unexplained_variance = 0
        for i, value in enumerate(data.columns.values):
            n = data[data.columns.values[i]].shape[0] - 100
            container[value] = 0
            unexplained_variance = 0
            for j in range(n):
                    unexplained_variance = unexplained_variance + (((float(data[value][j]) - data_mean_float[i]) ** 2) / (data_frame_size - k))
                # print('value is {} j is {} unexplained_variance is {}'.format(value, j, unexplained_variance))
            container[value] = unexplained_variance
        for key, value in container.items():
            print(key, value)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    user_input = input('please provide file name of full file path: ')
    data, labels, data_without_labels = load_data(user_input)
    f_test_for_iris(data_without_labels, labels)
