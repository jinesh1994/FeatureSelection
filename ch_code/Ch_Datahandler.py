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