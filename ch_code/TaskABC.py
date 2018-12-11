import numpy as np
import pandas as pd

from Ch_Datahandler import load_dataframe, store_data
from algos import FTest, SVM

# Task A
filename = '../GenomeTrainXY.txt'
# filename = '../ATNTFaceImages400.txt'
# filename = '../HandWrittenLetters.txt'

train_data_frame = load_dataframe(filename)

train_Y = train_data_frame.iloc[0, :]
train_X = train_data_frame.iloc[1:, :]
# print(train_data_frame)

labels, f_test_data = FTest.calculate(train_X=train_X, train_Y=train_Y)

top_n = 100

top_n_f_data = sorted(f_test_data, key=lambda x: x['F-score'], reverse=True)[:top_n]

[print('line_number:', f['line_number'], '  F-score:', f['F-score']) for f in top_n_f_data]

updated_train_X = [f['data'] for f in top_n_f_data]
updated_train_Y = labels

fname = 'f_sorted_output.txt'
updated_train_XY_df = store_data(filename=fname, X=updated_train_X, Y=updated_train_Y)
train_X_df = pd.DataFrame(updated_train_X)

# Task B
# a: SVM linear kernel
train_data_frame = load_dataframe(filename)
SVM_classifier = SVM()
SVM_classifier.fit(train_X_df.T, updated_train_Y)

# b: linear regression
# TODO

# c: KNN (k=3)
# TODO

# d: centroid method
# TODO

# Task C
test_filename = "../GenomeTestX.txt"
test_X_df = load_dataframe(test_filename)
test_X = test_X_df.iloc[:100, :]
predicted_data = SVM_classifier.predict(test_X.T)
print(predicted_data)
