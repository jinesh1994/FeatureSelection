import numpy as np
import pandas as pd

from Ch_Datahandler import load_dataframe, store_data
from algos import FTest


# Task A
filename = '../GenomeTrainXY.txt'
# filename = '../ATNTFaceImages400.txt'
# filename = '../HandWrittenLetters.txt'

train_data_frame = load_dataframe(filename)

train_Y = train_data_frame.iloc[0,:]
train_X = train_data_frame.iloc[1:,:]
# print(train_data_frame)

labels, f_test_data = FTest.calculate(train_X=train_X, train_Y=train_Y)

top_n = 100

top_n_f_data = sorted(f_test_data, key=lambda x: x['F-score'], reverse=True)[:top_n]

[print('line_number:', f['line_number'], '  F-score:', f['F-score']) for f in top_n_f_data]

updated_train_X = [f['data'] for f in top_n_f_data]
updated_train_Y = labels

fname = 'f_sorted_output.txt'
updated_train_XY_df = store_data(filename=fname, X=updated_train_X, Y=updated_train_Y)


# Task B
# a: SVM linear kernel
# TODO

# b: linear regression
# TODO

# c: KNN (k=3)
# TODO

# d: centroid method
# TODO

# Task C
# TODO