import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import median_filter
from scipy.integrate import solve_ivp


def ODE(P, F, time, h=0):
    A1 = 0.0154          # Area of tank in m^2
    A2 = 4.91e-4         # Area of drain pipe in m^2
    k_pump = 1  
    g = 9.81  
    k_opening = 1        # Fixed within range 2 to 8
    q_leak = 1.05e-5

    # t_vals = np.arange(0, time_duration, dt)
    h_vals = np.zeros_like(time)

    dt = time[1] - time[0]
    
    for i in range(len(time)):
        h = np.sqrt(h**2)
        dh_dt = (k_pump/A1)*P[i] - F[i]*k_opening*(A2/A1) * np.sqrt(2*g*h) - q_leak/A1
        h = h+dh_dt*dt
        h_vals[i] = h

    return time, h_vals


class Experiment():
    def __init__(self, csv_name_, set_values_vs_time = None):
        self.filename = csv_name_
        #use this line if file was originally .mf4
        data = pd.read_csv( self.filename, skiprows=29, usecols=[0, 1, 2, 3], names=['Time', 'Height', 'Pump', 'Valve'])  ##### !!!!!!!! Seeing as this was originally a .mf4 the columns on this .csv file look different to other csv files
        # use this line if file was orignially .csv
        # data = pd.read_csv( self.filename, skiprows=29, usecols=[1, 2, 3, 4], names=['Time', 'Height', 'Pump', 'Valve'])  ##### !!!!!!!! use this line for other .csv file (there is a 0 in the first columns hence the column shift)
        data = data.dropna()
        data = data.apply(pd.to_numeric, errors='coerce')
        for column in data.select_dtypes(include=[np.number]).columns:
            data[column + '_filtered'] = median_filter(data[column], size=5)
        self.data = data
        pass


file_path = r"Project\data\16_03_experiments\const_inflow_005.csv" # enter your file path here!!!!!
data = Experiment(file_path)
# print(data.data.head())


P = data.data['Pump']*0.1 #multiplying by 0.1 to change units
F = data.data['Valve']
# h0 = data.data['Height'].iloc[0]
h0 = 0 #since the value from data is negative it doesnt make sence and issues arrise in the model later, hence int height is 0
# time_duration = data.data['Time'].iloc[-1] ### dont need cuz it was changed to use time values extracted from experiment instead
time = data.data['Time']
print(data.data.head())

# print(f"int height = {h0}")
# print(f"final time = {time_duration}")

# Run the simulation
time_model, height_model = ODE(P, F, time)
height_in_mm_model = height_model*10  # Convert to mm

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(time_model, height_in_mm_model, label="Model Predicted Height", color='b', linestyle='dashed')
plt.plot(data.data['Time'].to_numpy(), data.data['Height'].to_numpy(), label = "Experimental Data")

# plt.scatter(time_exp, height_exp_data_mm, label="Experimental Data", color='r', marker='o', s=10)
# plt.plot(time_exp, height_exp_data_mm, label='experiment data', color = 'r')

plt.xlabel("Time [s]", weight= 'bold')
plt.ylabel("Water Height [mm]", weight = 'bold')
plt.title("Comparison of Model and Experimental Water Levels")
plt.legend()
plt.minorticks_on()
plt.grid(which ='major', linewidth = 0.5)
plt.grid(which = 'minor', linewidth = 0.2)
plt.show()