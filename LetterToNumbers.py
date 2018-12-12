import pandas as pd
# for char in 'jenil'.lower():
#     print(ord(char) - 96)


def letter_to_number(text):
    nums = [str(ord(x) - 96) for x in text.lower() if x >= 'a' and x <= 'z']
    return " ".join(nums)


def number_to_letter(classified_data):
    labels = []
    for data in classified_data:
        labels.append(chr(int(data) + 96))
    return labels


# print(letter_to_number('jenil'))
# print(number_to_letter([10, 5, 14, 9, 12]))


def splitData2TestTrain(filename_or_dataframe, total_number_per_class,  total_test_instances, test_first=True):
    """Takes filename, total_number_per_class and total_test_instances

    Assumes class labels is 1st row of file

    Returns:
    
    """
    if isinstance(filename_or_dataframe, str):
        dataframe = pd.read_csv(filename_or_dataframe, sep=",", header=None)
    else:
        dataframe = filename_or_dataframe

    test_X = [] # Test data without labels
    test_Y = [] # Test data labels only
    train_X = [] # Training data without labels
    train_Y = [] # Training data labels only
    
    if test_first:
        test_instance_count = dict()
        for col_number in dataframe.columns:
            val = dataframe[col_number].values
            # print(val[1:])
            label = val[0]
            current_count = test_instance_count.get(label, 0)
            if current_count<total_test_instances:
                # Add to test
                test_Y.append(label)
                test_X.append(val[1:])
            else:
                # Add to training
                train_Y.append(label)
                train_X.append(val[1:])
            current_count +=1
            test_instance_count[label] = current_count
    else:
        train_instance_count = dict()
        total_train_instances = total_number_per_class - total_test_instances
        for col_number in dataframe.columns:
            val = dataframe[col_number].values
            # print(val[1:])
            label = val[0]
            current_count = train_instance_count.get(label, 0)
            if current_count<total_train_instances:
                # Add to training
                train_Y.append(label)
                train_X.append(val[1:])
            else:
                # Add to test
                test_Y.append(label)
                test_X.append(val[1:])
            current_count +=1
            train_instance_count[label] = current_count


    return pandas.DataFrame(train_X).T, train_Y, pandas.DataFrame(test_X).T, test_Y
