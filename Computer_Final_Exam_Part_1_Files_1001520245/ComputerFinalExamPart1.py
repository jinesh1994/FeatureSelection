import pandas as pd
from Task_1_2_3 import k_means_algo


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
        # train_data_with_labels.to_csv('train_data_for_quiz_2.csv', sep=',', header=None, index=False)
        # test_data_with_labels.to_csv('test_data_for_quiz_2.csv', sep=',', header=None, index=False)
        return train_X, train_Y, test_X, test_Y, train_data_with_labels, test_data_with_labels
    except Exception as e:
        print(e)


# Task A
class_ids = letter_to_number('nilsaj')
student_id = '360745'
for i in student_id:
    class_ids.append(int(i))
# next line will give you data which has 12 classes.
data_frame = pickDataClass('HandWrittenLetters.txt', class_ids)
data_frame.to_csv('data_of_task_A.txt', sep=',', header=None, index=False)

# Task B,C,D,E
data_frame = data_frame.T
data_frame_without_label = data_frame.iloc[1:, :]
# labels are in columns which has size (100,0)
labels = data_frame.transpose()[0].values
# return data_frame, labels, data_frame_without_label
k_means_algo(data_frame_without_label, labels, 12)
