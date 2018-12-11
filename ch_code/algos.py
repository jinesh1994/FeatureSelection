import pandas as pd
import numpy as np
from sklearn.svm import SVC

class FTest(object):

    @staticmethod
    def calculate(train_Y, train_X):
        train_Y_set = set(train_Y)

        feature_statistics = list()

        for index, feature in enumerate(train_X.values):
            
            feature_averages = {k:{'sum':0, 'n':0, 'avg':0} for k in train_Y_set}
            
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

            # feature_statistics.append(
            #     {
            #         'index': index,
            #         'label_wise_avg': feature_averages,
            #         'label_wise_var': feature_variances,
            #         'overall_avg': overall_avg
            #     }
            # )

            F_num = sum([avgs['n'] * ((avgs['avg']-overall_avg) ** 2) for avgs in feature_averages.values()]) / (len(train_Y_set)-1)

            F_den = sum([(var['n']-1)*var['var'] for var in feature_variances.values()])/ (len(train_Y) - len(train_Y_set))

            if F_den>0:
                f_score = F_num/F_den
            else:
                f_score = float('inf')

            feature_statistics.append(
                {
                    'index': index,
                    'line_number': index+2,
                    'F-score':f_score,
                    'data': feature
                }
            )

        return list(train_Y), feature_statistics


class SVM(object):

    def __init__(self, kernel='rbf'):
        self._svc = SVC(kernel=kernel)
        # kernel : string, optional (default=’rbf’)
        # Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples)
        
    def fit(self, training_data_wo_label, training_labels):
        self._svc.fit(training_data_wo_label, training_labels)
    
    def predict(self, test_data):
        """Returns a list of labels it predicts
        """
        return self._svc.predict(test_data)
    
    def score(self, data, label):
        """
            data: Test Data
            label: Test Label/prediction
        """
        return self._svc.score(data, label)

class LinearRegressionMethod(object):

    @classmethod
    def calulate_accuracy(cls, training_dataframe, test_dataframe):
        N_train = len(training_dataframe.columns)
        N_test = len(test_dataframe.columns)
        Xtrain = training_dataframe.iloc[1:]
        Xtest = test_dataframe.iloc[1:]

        unique_training_labels = set(training_dataframe.iloc[0])
        indicator_values = dict()
        unique_training_labels_count = len(unique_training_labels)
        for i, label in enumerate(unique_training_labels):
            indicator_values[label] = [0] * unique_training_labels_count # [0,0,0,0,0]
            indicator_values[label][i] = 1

        Ytrain_list = list()
        for label in training_dataframe.iloc[0]:
            Ytrain_list.append(indicator_values[label])

        Ytrain = pd.DataFrame(Ytrain_list).T

        Ytest = test_dataframe.iloc[0]


        # The following is a Python code for Linear Regresion


        A_train = np.ones((1,N_train))    # N_train : number of training instance
        A_test = np.ones((1,N_test))      # N_test  : number of test instance
        Xtrain_padding = np.row_stack((Xtrain,A_train))
        Xtest_padding = np.row_stack((Xtest,A_test))

        '''computing the regression coefficients'''
        B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain.T)   # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix
        Ytest_padding = np.dot(B_padding.T,Xtest_padding)
        Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
        err_test_padding = Ytest - Ytest_padding_argmax
        TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100
        return TestingAccuracy_padding

