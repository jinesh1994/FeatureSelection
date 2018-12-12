import pandas as pd

def load_dataframe(file_path):

    data_frame = pd.read_csv(file_path, header=None, sep=",")

    return data_frame

def to_data_frame(X, Y):
    """
    X is a Data Frame
    Y is a list of labels
    
    """
    # Getting values row wise
    if not isinstance(X, list):
        temp_X = X.values
    else:
        temp_X = X
    
    if not isinstance(Y, list):
        temp_Y = list(Y)
    else:
        temp_Y = Y
    
    temp_XY = list()
    temp_XY.append(temp_Y)
    for array in temp_X:
        temp_XY.append(list(array))
    
    return pd.DataFrame(temp_XY)

def store_data(filename, X, Y):
    """
    X is a Data Frame
    Y is a list of labels
    
    """
    df = to_data_frame(X, Y)
    df.to_csv(filename,header=False, index=False, index_label=False)
    return df


def pick_data_class(filename, labels):
    """Takes filename and labels needed to pick

    Assumes class labels is 1st row of file

    Returns:
    Dataframe of results list.
    """

    dataframe = pd.read_csv(filename, sep=",", header=None)
    dataframe_T = dataframe.T
    dataframe_T_values = dataframe_T.values

    result_list = list()

    for each in dataframe_T_values:
        each_list = list(each)
        # print(each_list[0])
        # print
        if each_list[0] in labels:
            result_list.append(each_list)
    
    return pd.DataFrame(result_list).T

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


    return pd.DataFrame(train_X).T, train_Y, pd.DataFrame(test_X).T, test_Y
