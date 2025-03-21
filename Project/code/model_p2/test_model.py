import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import median_filter
import numpy as np

def ODE(P, F, time, h=0):
    A1 = 0.0154          # Area of tank in [m^2]
    A2 = 4.91e-4         # Area of drain pipe in [m^2]
    A3 = 0.000075        # Area of the leak hole [m^2]
    k_pump = 1.0         # Pump coefficient
    g = 9.8140           # Gravitational acceleration [m/s^2]
    k_opening = 1.005    # Valve opening coefficient
    q_leak = 0.000000002 # Leakage flow rate [m^3/s]

    h_vals = np.zeros_like(time)
    dt = time[1] - time[0]
   
    for i in range(len(time)):
        h = np.sqrt(h**2)  # Ensure h is not negative
        dh_dt = (k_pump/A1)*P[i] - F[i]*k_opening*(A2/A1) * np.sqrt(2*g*h) - F[i]*k_opening*(A3/A1) * np.sqrt(2*g*h) 
        h = h + dh_dt*dt
        h_vals[i] = h

    return time, h_vals

class Experiment():
    def __init__(self, csv_name_, set_values_vs_time=None):
        self.filename = csv_name_
        data = pd.read_csv(self.filename, skiprows=29, usecols=[0, 1, 2, 3], names=['Time', 'Height', 'Pump', 'Valve'])
        data = data.dropna()
        data = data.apply(pd.to_numeric, errors='coerce')
        for column in data.select_dtypes(include=[np.number]).columns:
            data[column + '_filtered'] = median_filter(data[column], size=5)
        self.data = data

file_path = r"Project\data\24_02_25\rec1_004.csv"  # Enter your file path here
data = Experiment(file_path)

P = data.data['Pump'] *0.1 # Multiplying by 0.1 to change units
F = data.data['Valve']
time = data.data['Time']

# Run the model
time_model, height_model = ODE(P, F, time)
height_in_mm_model = height_model * 10  # Convert to mm

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(time_model, height_in_mm_model, label="Model Predicted Height", color='b', linestyle='dashed')
plt.plot(data.data['Time'].to_numpy(), data.data['Height'].to_numpy(), label="Experimental Data", color='red')
plt.xlabel("Time (s)", weight='bold')
plt.ylabel("Water Height (mm)", weight='bold')
plt.title("Comparison of Model and Experimental Water Levels")
plt.legend()
plt.minorticks_on()
plt.grid(which='major', linewidth=0.5)
plt.grid(which='minor', linewidth=0.2)
plt.show()