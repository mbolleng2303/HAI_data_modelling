
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
import matplotlib.cm as cm
"""file_name = 'Excel_tijdslijn_maatregelen.xlsx'
file_name = 'Tr_data.csv'
dfs = pd.read_csv(file_name, delimiter=';')"""

class HA_Data:

    def __init__(self, outcome="Weekly COVID-19 Cases in Belgium"):
        self.path = './'
        self.file_name = 'Tr_data.csv'
        self.data = pd.read_csv(self.path+self.file_name)
        self.outcome = outcome
        self.data.info()
        self.data.head()

    def explore_dataset(self):
        data = self.data
        # plotpair
        plt.figure()
        seaborn.pairplot(data)
        plt.title('Data pairs distribution')
        plt.savefig(self.path + 'data_pairplot.png')

        plt.figure(figsize=(12, 8))
        corr = data.corr()
        mask = np.tri(*corr.shape).T
        seaborn.heatmap(corr.abs(), mask=mask, annot=True)
        b, t = plt.ylim()
        b += 0.5
        t -= 0.5
        plt.ylim(b, t)
        plt.title('Correlation between data')
        plt.savefig(self.path + 'data_correlation.png')

        # see impact to the target
        plt.figure()
        n_fts = len(data.columns)
        colors = cm.rainbow(np.linspace(0, 1, n_fts))
        data.drop('outcome', axis=1).corrwith(data.outcome).sort_values(ascending=True).plot(kind='barh',
                                                                                             color=colors,
                                                                                             figsize=(12, 8))
        plt.title('Correlation to Target (outcome)')
        plt.savefig(self.path + 'data_corr2target.png')
        plt.figure()