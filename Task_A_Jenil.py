import pandas as pd
import sys
import os
from contextlib import redirect_stdout
from itertools import islice


def load_data(file_path):
    try:
        # for data which has class numbers(labels) in 1st row.
        if 'ris' not in file_path:
            data_frame = pd.read_csv(file_path, header=None, sep=",")
            data_frame_T = data_frame.T
            class_numbers = data_frame_T[0]
            data_frame_T = data_frame_T.iloc[:, 1:].reset_index(drop=True)
            data_frame_T.index = class_numbers.index
            new_data_frame = pd.concat([data_frame_T, class_numbers], axis=1)
            new_data_frame = new_data_frame.rename(columns={new_data_frame.columns[-1]: "class numbers"})
            print('Data has been cleaned and rearranged for F-test algorithm')
            return new_data_frame, class_numbers, data_frame_T
        elif 'ris' in file_path:
            data_frame = pd.read_csv(file_path, header=None, sep=",")
            class_numbers = data_frame.iloc[:, -1]
            only_data_frame = data_frame.iloc[:, :-1]
            only_data_frame.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
            print('Data has been cleaned')
            data_frame.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class numbers']
            return data_frame, class_numbers, only_data_frame
    except Exception as e:
        print(e)


def f_test(data, label_values):
    try:
        # column wise mean of the dataframe.
        if os.path.isfile('jenil_result.txt'):
            os.remove('jenil_result.txt')
        data_mean = data.iloc[:, :-1].mean()
        total_class_numbers = set(data.iloc[:, -1])
        data_mean_float = []
        for i in data_mean:
            data_mean_float.append(float(i))
        container_for_groups = {}
        for i, value in enumerate(data.iloc[:, :-1].columns.values):
            variance_values = []
            average_values = []
            data_frame_size = []
            for j in total_class_numbers:
                class_data_frame = pd.DataFrame(data.loc[data['class numbers'] == j][value])
                data_frame_size.append(class_data_frame.size)
                mean_of_class = float(class_data_frame.mean())
                average_values.append(mean_of_class)

                new_data_frame_with_subtracting_mean = pd.DataFrame(class_data_frame[value] - mean_of_class)
                square_of_data_frame = pd.DataFrame(new_data_frame_with_subtracting_mean[value] ** 2)

                variance = float(square_of_data_frame[value].sum() / float(int(class_data_frame.size) - 1))
                variance_values.append(variance)

            mean_of_whole_column = data_mean_float[i]

            data_frame = pd.DataFrame(data[value])
            value_1 = 0
            for j in range(len(total_class_numbers)):
                value_1 = value_1 + int(data_frame_size[j]) * ((average_values[j] - mean_of_whole_column) ** 2)
            numerator = value_1 / (len(total_class_numbers) - 1)
            value_2 = 0
            for k in range(len(variance_values)):
                size = int(data_frame_size[k]) - 1
                value_2 = value_2 + (size * variance_values[k])
            denominator = float(value_2) / (len(data_frame) - len(total_class_numbers))
            if not numerator or not denominator:
                final_result = float(0)
            if numerator and denominator:
                final_result = numerator / denominator
            print('{} f={}'.format(value, final_result))
            container_for_groups[value] = final_result
            with open('jenil_result.txt', 'a') as f:
                with redirect_stdout(f):
                    print(value, final_result)
        container_for_groups_sorted_keys = sorted(container_for_groups.items(), key=lambda t: t[1], reverse=True)
        select_rows = int(input('\nplease select rows you want: '))
        selected_rows = take(select_rows, container_for_groups_sorted_keys)
        if os.path.isfile('jenil_output_result.txt'):
            os.remove('jenil_output_result.txt')
        for i in selected_rows:
            file = open('jenil_output_result.txt', 'a')
            file.write('\n')
            file.write(i[0])
            file.write('\n')
            file.write(str(i[1]))
        # file.close()
        get_top_data_as_per_user(data, container_for_groups_sorted_keys, select_rows, label_values)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def get_top_data_as_per_user(all_data, dictionary, input_value, labels):
    try:
        data_frame = pd.DataFrame()
        data_frame.insert(0, 0, list(labels.values))
        for i in range(0, input_value):
            temp = all_data[dictionary[i][0]]
            data_frame.insert(i, i + 1, temp)
        new_data_frame = pd.DataFrame()
        for j in sorted(data_frame.columns.values):
            new_data_frame.insert(int(j), int(j), data_frame[j])
        new_data_frame.T.to_csv('jenil_top_{}_data.csv'.format(input_value), sep=',', header=None, index=False)
        print('jenil_top_{}_data.csv has been generated'.format(input_value))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    user_input = input('please provide file name of full file path: ')
    data, labels, data_without_labels = load_data(user_input)
    f_test(data=data, label_values=labels)
