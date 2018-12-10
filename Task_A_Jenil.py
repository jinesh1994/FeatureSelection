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
            data_frame.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class numbers']
            return data_frame, class_numbers, only_data_frame
    except Exception as e:
        print(e)


def f_test_for_iris(data):
    try:
        # column wise mean of the dataframe.
        # TODO start for loop for groups. here we have 3 groups and 4 features.
        data_mean = data.iloc[:, :-1].mean()
        total_class_numbers = set(data.iloc[:, -1])
        # data_frame_size = int(data.iloc[:, :-1].size)
        data_mean_float = []
        for i in data_mean:
            data_mean_float.append(float(i))
        # explained_variance = 0
        k = len(data.columns.values)
        container_for_groups = {}
        # overall_data_mean = np.mean(data_mean_float)
        for i, value in enumerate(data.iloc[:, :-1].columns.values):
            container_for_groups[value] = []
            variance_values = []
            average_values = []
            data_frame_size = []
            for j in total_class_numbers:
                class_data_frame = pd.DataFrame(data.loc[data['class numbers'] == j][value])
                data_frame_size.append(class_data_frame.size)
                mean_of_class = float(class_data_frame.mean())
                average_values.append(mean_of_class)
                container_for_groups[value].append('mean of {} {}st/nd part is {} '.format(value, j, mean_of_class))
                new_data_frame_with_subtracting_mean = pd.DataFrame(class_data_frame[value] - mean_of_class)
                square_of_data_frame = pd.DataFrame(new_data_frame_with_subtracting_mean[value] ** 2)
                # TODO divide next line by 49. (class data frame size - 1)
                variance = float(square_of_data_frame[value].sum() / float(int(class_data_frame.size) - 1))
                variance_values.append(variance)
                # float(sum_of_squares_within_groups + total)
                container_for_groups[value].append('sum of squares within groups is {} after {}th/nd loop '.format(variance_values, j))
            mean_of_whole_column = data_mean_float[i]
            container_for_groups[value].append('mean of {}th/nd column is{}'.format(i, mean_of_whole_column))
            data_frame = pd.DataFrame(data[value])
            # zip_of_average_and_variance = zip(average_values, variance_values)
            # f = 0

            value_1 = 0
            for j in range(len(total_class_numbers)):
                value_1 = value_1 + int(data_frame_size[j]) * ((average_values[j] - mean_of_whole_column) ** 2)
            numer = value_1 / (len(total_class_numbers) - 1)
            # print('container is', numer)

            value_2 = 0
            for k in range(len(variance_values)):
                value_2 = value_2 + int(data_frame_size[k]) - 1 * variance_values[k]
            den = float(value_2) / (int(data_frame.size) - len(total_class_numbers))
            final_result = numer / den
            print('{} f={}'.format(value, final_result))
            # for average, variance in zip(average_values, variance_values):
            #     temp = class_data_frame.size()
            # data_frame_after_subtracting_mean_of_whole_column = pd.DataFrame(data_frame[value] - mean_of_whole_column)
            # square_of_data_frame_after_subtracting_mean_of_whole_column = pd.DataFrame(data_frame_after_subtracting_mean_of_whole_column[value] ** 2)
            # sum_of_square_of_data_frame_after_subtracting_mean_of_whole_column = float(square_of_data_frame_after_subtracting_mean_of_whole_column[value].sum())
            # sum_of_square_between_groups = float(sum_of_square_of_data_frame_after_subtracting_mean_of_whole_column - sum_of_squares_within_groups)
            # container_for_groups[value].append('sum of square between groups is {}'.format(sum_of_square_between_groups))
            # sum_of_square_between_groups_divide_by_degrees_of_freedom = float(sum_of_square_between_groups / len(total_class_numbers) - 1)
            # container_for_groups[value].append('sum of square between group divide by degrees of freedom is {}'.format(sum_of_square_between_groups_divide_by_degrees_of_freedom))
            # degrees_of_freedom_for_within_groups = float(data_frame.size - len(total_class_numbers))
            # sum_of_square_within_groups_divide_by_degrees_of_freedom = float(sum_of_squares_within_groups / degrees_of_freedom_for_within_groups)
            # container_for_groups[value].append('sum of square within groups divide by degrees of freedom is {}'.format(sum_of_square_within_groups_divide_by_degrees_of_freedom))
            # f_test_result = float(sum_of_square_between_groups_divide_by_degrees_of_freedom / sum_of_square_within_groups_divide_by_degrees_of_freedom)

            # print('{} f= {}'.format(value, f_test_result))
            # for key, values in container_for_groups.items():
            #     print(key, values)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    user_input = input('please provide file name of full file path: ')
    data, labels, data_without_labels = load_data(user_input)
    f_test_for_iris(data=data)
