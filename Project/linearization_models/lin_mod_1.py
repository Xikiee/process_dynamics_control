import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sympy as sp
import pandas as pd
from scipy.ndimage import median_filter
import math 


def system_step_response(num1, den1, step_size, exp_data):
    sys = ctrl.TransferFunction(num1,den1)

    t_end = math.ceil(data.data["Time"].iloc[-1])
    t = np.linspace(0,t_end,t_end*2)

    h0 = exp_data.data['Height'].iloc[0]
    t,y = ctrl.step_response(sys,t)
    y_new = y*step_size + h0


    plt.plot(t,y_new, label = 'Model Height')   #plotting model
    plt.plot(data.data['Time'].to_numpy(), data.data['Height'].to_numpy(), label = 'Experiment Height') #experiment 
    plt.xlabel('Time (s)')
    plt.ylabel('y(t)')
    plt.grid(which = 'both', linewidth = 0.5)
    plt.legend()
    plt.show()



a1 = 0.0154  # Area of tank in m^2
a2 = 4.91e-4  # Area of drain pipe in m^2
k_pump = 1.1  
g = 9.81  
k_opening = 5  # Fixed within range 2 to 8
a3 = 0.00015


f = 0.3                  # change f for the flow rate in question
p = 0.01                 # this is the change in the pump value (initial value was 0.06 and was changed to 0.07)
h_s = 350                # steady state height

q1= k_pump/a1
q2= ((a2/a1)+(a3/a1)) * k_opening * f * np.sqrt(2*g)


num = [(q1*2*np.sqrt(h_s))/q2]
den = [(2*np.sqrt(h_s))/q2,1]


#import the data from the experiment 
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

name = 'oliver'
data = Experiment('project /processed_data/step_change_experiment_data/exp_5.csv', name)


system_step_response(num,den,p, data)




#yea idk this code works really well for exp 6, but then for other experiments its kinda shit so idk whats exactly up with it