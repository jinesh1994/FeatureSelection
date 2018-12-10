import numpy as np
import pandas as pd

from Ch_Datahandler import load_dataframe
from algos import FTest


filename = '../Iris.txt'

train_data_frame = load_dataframe(filename)
# print('train_data_frame1', type(train_data_frame))
train_data_frame = pd.DataFrame(np.fliplr(train_data_frame))
# print('train_data_frame2', train_data_frame)

train_data_frame = train_data_frame.T
# print('train_data_frame3', train_data_frame.values[0])

# train_XY_values = train_data_frame
# print('train_XY_values1', type(train_XY_values))

train_Y = train_data_frame.iloc[0,:]
train_X = train_data_frame.iloc[1:,:]
# print(train_data_frame)
labels, f_test_data = FTest.calculate(train_X=train_X, train_Y=train_Y)
[print(f) for f in f_test_data]