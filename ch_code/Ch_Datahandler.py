import pandas as pd

def load_dataframe(file_path, transpose=False):

    data_frame = pd.read_csv(file_path, header=None, sep=",")

    if transpose:
        return data_frame
    
    return data_frame
