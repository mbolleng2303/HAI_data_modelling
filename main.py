import os
import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from Data.Ha_Data import HA_Data
import pandas as pd
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
import matplotlib.cm as cm
'''import warnings
warnings.filterwarnings('ignore')'''
import os
import numpy as np
import pickle
import pymc3 as pm
import time

def explore_dataset(data, save_path, outcome='cases_BE'):
    # plotpair
    seaborn.pairplot(data)
    plt.title('Data pairs distribution')
    plt.show()
    plt.savefig(save_path+'data_pairplot.png')
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    mask = np.tri(*corr.shape).T
    seaborn.heatmap(corr.abs(), mask=mask, annot=True)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.title('Correlation between data')
    plt.savefig(save_path+'data_correlation.png')

    # see impact to the target
    plt.figure()
    n_fts = len(data.columns)
    colors = cm.rainbow(np.linspace(0, 1, n_fts))
    data.drop(outcome, axis=1).corrwith(data[outcome]).sort_values(ascending=True).plot(kind='barh',
                                                                                         color=colors,
                                                                                         figsize=(12, 8))
    plt.title('Correlation to Target {}'.format(outcome))
    plt.savefig(save_path+'data_corr2target.png')


def change_to_ordered(A,B,y):
    for i in range(len(y)):
        if y[i] < A:
            y[i] = 0
        elif y[i] > B:
            y[i] = 2
        else:
            y[i] = 1
    return y

def make_plot(model, trace, model_path, var_names = None):

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(model.basic_RVs)

    az.plot_trace(trace, var_names=var_names, combined=False)
    plt.savefig(model_path + 'trace')

    az.summary(trace).to_csv(model_path + 'trace.csv')
    print(az.summary(trace))

    az.plot_pair(trace, point_estimate="median", var_names=var_names)
    plt.savefig(model_path + 'plot_pair')

    az.plot_energy(trace)
    plt.savefig(model_path + 'energy')

    az.plot_posterior(trace, var_names=var_names)
    plt.savefig(model_path + 'posterior')

    az.plot_forest(trace, var_names=var_names)
    '''for j, (y_tick, frac_j) in enumerate(zip(plt.gca().get_yticks(), reversed(true_coeff))):
        plt.vlines(frac_j, ymin=y_tick - 0.45, ymax=y_tick + 0.45, color="black", linestyle="--")'''
    plt.savefig(model_path + 'forest', dpi=100)

if __name__ == '__main__':
    import pandas as pd
    data = 'Tr_data_3'
    outcome = 'cases_BE_3'
    save_path = 'Save/' + data + '_' + outcome + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = 'Data/'+data+'.csv'
    df = pd.read_csv(file_name, delimiter=';')
    df = df[['cases_BE', 'cases_BE_1', 'cases_BE_2', 'cases_BE_3',
            'NPI_BE', 'NPI_BE_1', 'NPI_BE_2', 'NPI_BE_3',
            'VOC', 'VOC_1', 'VOC_2', 'VOC_3',
            'Year', 'Week']
    ]
    SEED = 42
    tune = 10
    chains = 1
    draws = 10
    nbr_classes = 3
    notice = 'full'
    model_path = "./result/ordered_" + time.strftime('%Hh%Mm_%m_%d_%y') + notice + '_' + outcome + '_' + str(
        draws) + '_' + str(tune) + '_' + str(chains) + '/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    np.random.seed(SEED)
    #df.drop(columns=['ND', 'NI', 'NP'])
    #explore_dataset(df, save_path, outcome=outcome)

    x = df.drop([outcome], axis=1).values
    y = df[outcome].values



    A = np.percentile(y, 33)
    B = np.percentile(y, 66)
    y = change_to_ordered(A, B, y)
    for i in range(4):
        x[:, i] = change_to_ordered(A, B, x[:, i])


    x=x[3:102,:]
    y=y[3:102]
    with pm.Model() as ordered_multinomial:
        '''cutpoints = pm.Normal(
            "cutpoints",
            0.0,
            100,
            transform=pm.distributions.transforms.ordered,
            shape=nbr_classes - 1,
            testval=np.array(true_coeff[x.shape[1]:x.shape[1] + nbr_classes]),
        )
        #  a = pm.Normal("intercept", mu=true_coeff[0], sigma=5)  # intercepts
        b = pm.Normal("age", mu=true_coeff[0], sigma=5)
        c = pm.Normal("gender", mu=true_coeff[1], sigma=5)
        d = pm.Normal("smoking", mu=true_coeff[2], sigma=5)
        e = pm.Normal("fever", mu=true_coeff[3], sigma=5)
        f = pm.Normal("vomiting", mu=true_coeff[4], sigma=5)'''

        cutpoints = pm.Normal(
            "cutpoints",
            0.0,
            100,
            transform=pm.distributions.transforms.ordered,
            shape=nbr_classes - 1,
            testval=np.arange(nbr_classes - 1),
        )
        #  a = pm.Normal("intercept", mu=0, sigma=100)  # intercepts
        b = pm.Normal("cases_BE", mu=0, sigma=100, shape=3)
        c = pm.Normal("NPI_BE", mu=0, sigma=100, shape=4)
        d = pm.Normal("VOC", mu=0, sigma=100, shape = 4)
        '''e = pm.Normal("Week", mu=0, sigma=100)
        f = pm.Normal("Year", mu=0, sigma=100)'''
        phi = b[0] * x[:, 0] + b[1] * x[:, 1] + b[2] * x[:, 2] +\
              c[0] * x[:, 3] + c[1] * x[:, 4] + c[2] * x[:, 5] + c[3] * x[:, 6]+ \
              d[0] * x[:, 7] + d[1] * x[:, 8] + d[2] * x[:, 9] + d[3] * x[:, 10]
        outcome = pm.OrderedLogistic("ordered_outcome", eta=phi, cutpoints=cutpoints, observed=y)
        trace_ordered_multinomial = pm.sample(draws=draws,
                                              tune=tune,
                                              chains=chains,
                                              cores=1,
                                              init='auto', progressbar=True)

        with open(model_path + 'model.pkl', 'wb') as buff:
            pickle.dump({'model': ordered_multinomial, 'trace': trace_ordered_multinomial}, buff)
        with open(model_path + 'model.pkl', 'rb') as buff:
            data0 = pickle.load(buff)
        model, trace = data0['model'], data0['trace']
        make_plot(model, trace,
                  model_path, var_names=['cases_BE', 'NPI_BE', 'VOC', 'cutpoints'])



