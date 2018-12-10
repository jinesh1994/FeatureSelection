import numpy as np
import pandas as pd

from Ch_Datahandler import load_dataframe


filename = 'Iris.txt'

train_data_frame = load_dataframe(filename, False)
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
train_Y_set = set(train_Y)

# train_Y = train_data_frame.iloc[:, -1]

feature_statistics = dict()

for index, feature in enumerate(train_X.values):
    
    feature_averages = {k:{'sum':0, 'n':0, 'avg':0} for k in train_Y_set}
    # print(feature_averages)
    # Label Wise
    # Avg
    for i, feature_val in enumerate(feature):
        label = train_Y[i]
        label_data = feature_averages[label]

        label_data['sum'] += feature_val
        label_data['n'] += 1
    
    # Cal avg
    for label, aggregates in feature_averages.items():
        aggregates['avg'] = aggregates['sum']/aggregates['n']
    
    # print(feature_averages)

    # Var
    feature_variances = {k:{'sum':0, 'n':0, 'var':0} for k in train_Y_set}
    
    for i, feature_val in enumerate(feature):
        label = train_Y[i]
        label_data = feature_variances[label]


        label_data['sum'] += ((feature_val-feature_averages[label]['avg']) ** 2)
        label_data['n'] += 1

    # Cal var
    for label, aggregates in feature_variances.items():
        aggregates['var'] = aggregates['sum']/(aggregates['n'] - 1)
    
    
    overall_avg = sum(feature)/len(feature)

    feature_statistics[index] = {
        'index': index,
        'label_wise_avg': feature_averages,
        'label_wise_var': feature_variances,
        'overall_avg': overall_avg
    }
    # print(feature_statistics)
    F_num = sum([avgs['n'] * ((avgs['avg']-overall_avg) ** 2) for avgs in feature_averages.values()]) / (len(train_Y_set)-1)

    # F_num = []
    # for ky, avgs in feature_averages.items():
    #     print(ky, 
    #         (avgs['avg']-overall_avg) ** 2
    #     )
    F_den = sum([(var['n']-1)*var['var'] for var in feature_variances.values()])/ (len(train_Y) - len(train_Y_set))

    print('feature_number', index,' F=',F_num/F_den)
# [print(f) for f in feature_statistics.values()]