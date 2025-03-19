import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ODE(P, F, h0, time_duration, dt=0.1):
    A1 = 0.0154  # Area of tank in m^2
    A2 = 4.91e-4  # Area of drain pipe in m^2
    k_pump = 1.1  
    g = 9.81  
    k_opening = 5  # Fixed within range 2 to 8
    h = h0  # Initial height in the tank
    
    t_vals = np.arange(0, time_duration, dt)
    h_vals = np.zeros_like(t_vals)
    
    for i, t in enumerate(t_vals):  # Integrating (using Euler method)
        dh_dt = (k_pump/A1) * P - (k_opening * F * (A2 / A1) * np.sqrt(2 * g * h))  # Equation from model finalization
        h = max(0, h + dh_dt * dt)  # Preventing negatives
        h_vals[i] = h
    return t_vals, h_vals

P = 0.01  # Pump setting
F = 0.0  # Fraction valve opening
h0 = 0.0  # Initial height in meters
time_duration = 50  # Duration for which this setting was kept
file_path = "const_outflow_90.csv" # CSV file containing experimental data
t_offset = 10

# Run the simulation
time_model, height_model = ODE(P, F, h0, time_duration)
height_in_mm_model = height_model*10  # Convert to mm

# Load experimental data
df = pd.read_csv(file_path)
time_exp = df.iloc[:, 0].values  # Time in column 1
height_exp_data_mm = df.iloc[:, 1].values  # Height in column 2

time_model = time_model + t_offset

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(time_model, height_in_mm_model, label="Model Predicted Height", color='b', linestyle='dashed')
plt.scatter(time_exp, height_exp_data_mm, label="Experimental Data", color='r', marker='o', s=10)
plt.xlabel("Time [s]")
plt.ylabel("Water Height [mm]")
plt.title("Comparison of Model and Experimental Water Levels")
plt.legend()
plt.grid()
plt.show()