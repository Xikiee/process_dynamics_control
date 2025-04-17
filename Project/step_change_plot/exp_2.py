import numpy as np
import scipy as sc
import pandas as pd
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

class Experiment():
    def __init__(self, csv_name_, set_values_vs_time = None):
        self.filename = csv_name_
        data = pd.read_csv( self.filename, skiprows=29, usecols=[1, 2, 3, 4], names=['Time', 'Height', 'Pump', 'Valve'])
        data = data.dropna()
        data = data.apply(pd.to_numeric, errors='coerce')
        for column in data.select_dtypes(include=[np.number]).columns:
            data[column + '_filtered'] = median_filter(data[column], size=5)
        self.data = data
        pass

def plotting(data,xlabel,ylabel):
    plt.plot(data.data['Time'].to_numpy(), data.data['Height'].to_numpy())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.minorticks_on()
    plt.grid(which = "both", linewidth = 0.5)
    plt.show()



name = 'Oliver'
data = Experiment('project /processed_data/step_change_experiment_data/exp_2.csv', name)
plotting(data,'time', 'height' )
# print(data.data.head())  
