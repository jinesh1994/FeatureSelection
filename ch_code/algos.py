

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
                f_score = 0.0

            feature_statistics.append(
                {
                    'line_number': index+1,
                    'F-score':f_score,
                    'data': feature
                }
            )

        return train_Y, feature_statistics