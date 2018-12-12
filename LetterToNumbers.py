import pandas as pd


# for char in 'jenil'.lower():
#     print(ord(char) - 96)


def letter_to_number(text):
    # nums = [str(ord(x) - 96) for x in text.lower() if x >= 'a' and x <= 'z']
    # return " ".join(nums)
    return [ord(character) - 96 for character in text.lower()]


def number_to_letter(classified_data):
    labels = []
    for data in classified_data:
        labels.append(chr(int(data) + 96))
    return labels


# Returns data row wise - 1st column has labels
def pickDataClass(filename, class_ids):
    load_file = pd.read_csv(filename, sep=",", header=None)
    load_file = load_file.transpose()
    result = []
    for i in class_ids:
        for j in load_file.values:
            if int(j[0]) == int(i):
                result.append(j)
    result = pd.DataFrame(result)
    return result


# print(letter_to_number('jenil'))
# print(number_to_letter([10, 5, 14, 9, 12]))


# def splitData2TestTrain(filename_or_dataframe, total_number_per_class, total_test_instances, test_first=True):
#     """Takes filename, total_number_per_class and total_test_instances
#
#     Assumes class labels is 1st row of file
#
#     Returns:
#
#     """
#     if isinstance(filename_or_dataframe, str):
#         dataframe = pd.read_csv(filename_or_dataframe, sep=",", header=None)
#     else:
#         dataframe = filename_or_dataframe
#
#     test_X = []  # Test data without labels
#     test_Y = []  # Test data labels only
#     train_X = []  # Training data without labels
#     train_Y = []  # Training data labels only
#
#     if test_first:
#         test_instance_count = dict()
#         for col_number in dataframe.columns:
#             val = dataframe[col_number].values
#             # print(val[1:])
#             label = val[0]
#             current_count = test_instance_count.get(label, 0)
#             if current_count < total_test_instances:
#                 # Add to test
#                 test_Y.append(label)
#                 test_X.append(val[1:])
#             else:
#                 # Add to training
#                 train_Y.append(label)
#                 train_X.append(val[1:])
#             current_count += 1
#             test_instance_count[label] = current_count
#     else:
#         train_instance_count = dict()
#         total_train_instances = total_number_per_class - total_test_instances
#         for col_number in dataframe.columns:
#             val = dataframe[col_number].values
#             # print(val[1:])
#             label = val[0]
#             current_count = train_instance_count.get(label, 0)
#             if current_count < total_train_instances:
#                 # Add to training
#                 train_Y.append(label)
#                 train_X.append(val[1:])
#             else:
#                 # Add to test
#                 test_Y.append(label)
#                 test_X.append(val[1:])
#             current_count += 1
#             train_instance_count[label] = current_count
#
#     return pd.DataFrame(train_X).T, train_Y, pd.DataFrame(test_X).T, test_Y


# filename: expects string name of file including full path OR
# passes dataframe in vertical format which is same as file format, i.e. 1st row has labels
def splitData2TestTrain(filename, number_per_class, test_instances, train_first=False):
    try:
        if '.txt' in str(filename):
            filename = pd.read_csv(filename, sep=",", header=None)
        else:
            filename = filename.transpose()
        # we know that this is dataframe.
        test_X = []  # Test data without labels
        test_Y = []  # Test data labels only
        train_X = []  # Training data without labels
        train_Y = []  # Training data labels only
        train_data_with_labels = []
        test_data_with_labels = []
        test_instance_count = dict()
        for col_number in filename.columns:
            val = filename[col_number].values
            label = val[0]
            current_count = test_instance_count.get(label, 0)
            if ((not train_first and current_count < test_instances) or (
                    train_first and current_count >= number_per_class - test_instances)):
                # Add to test
                test_Y.append(chr(label + 96))
                test_X.append(val[1:])
                test_data_with_labels.append(val)
            else:
                # Add to training
                train_Y.append(chr(label + 96))
                train_X.append(val[1:])
                train_data_with_labels.append(val)
            current_count += 1
            test_instance_count[label] = current_count
        train_X = pd.DataFrame(train_X)
        test_X = pd.DataFrame(test_X)
        train_data_with_labels = pd.DataFrame(train_data_with_labels)
        test_data_with_labels = pd.DataFrame(test_data_with_labels)
        return train_X, train_Y, test_X, test_Y, train_data_with_labels, test_data_with_labels
    except Exception as e:
        print(e)


# class_ids = letter_to_number('jenil')
# datas = pickDataClass('HandWrittenLetters.txt', class_ids)
# train_data_set_without_labels, train_y, test_data_set_without_labels, test_y, train_data_with_labels, test_data_with_labels = splitData2TestTrain(
#     datas, 39, 19, True)
# print(train_data_set_without_labels, train_y, test_data_set_without_labels, test_y, train_data_with_labels,
#       test_data_with_labels)
