import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import median_filter
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Define the ODE function for solve_ivp
def tank_ode(t, h, P_interp, F_interp):
    A1 = 0.0154          # Area of tank in m^2
    A2 = 4.91e-4         # Area of drain pipe in m^2
    k_pump = 1  
    g = 9.81  
    k_opening = 1        # Fixed within range 2 to 8
    q_leak = 1.05e-5

    # Interpolate P and F at the current time t
    P = P_interp(t)
    F = F_interp(t)

    # ODE equation
    dh_dt = (k_pump / A1) * P - F * k_opening * (A2 / A1) * np.sqrt(2 * g * h) - q_leak / A1
    return dh_dt


def ODE(P, F, time, h0=0):
    # Create interpolation functions for P and F
    P_interp = interp1d(time, P, kind='linear', fill_value="extrapolate")
    F_interp = interp1d(time, F, kind='linear', fill_value="extrapolate")

    # Solve the ODE using solve_ivp
    sol = solve_ivp(tank_ode, [time[0], time[-1]], [h0], args=(P_interp, F_interp), t_eval=time, method='RK45')

    # Extract the solution
    h_vals = sol.y[0]
    return sol.t, h_vals


class Experiment():
    def __init__(self, csv_name_, set_values_vs_time=None):
        self.filename = csv_name_
        # Use this line if file was originally .mf4
        data = pd.read_csv(self.filename, skiprows=29, usecols=[0, 1, 2, 3], names=['Time', 'Height', 'Pump', 'Valve'])
        # Use this line if file was originally .csv
        # data = pd.read_csv(self.filename, skiprows=29, usecols=[1, 2, 3, 4], names=['Time', 'Height', 'Pump', 'Valve'])
        data = data.dropna()
        data = data.apply(pd.to_numeric, errors='coerce')
        for column in data.select_dtypes(include=[np.number]).columns:
            data[column + '_filtered'] = median_filter(data[column], size=5)
        self.data = data


# Load data
file_path = r"Project\data\24_02_25\rec1_002.csv"  # Enter your file path here!!!!!
data = Experiment(file_path)

# Extract data
P = data.data['Pump'] * 0.1  # Multiply by 0.1 to change units
F = data.data['Valve']
time = data.data['Time']
h0 = 0  # Initial height

# Run the simulation
t_span = [data.data['Time']]
time_model, height_model = ODE(P, F, time, h0)
height_in_mm_model = height_model * 1000  # Convert to mm

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(time_model, height_in_mm_model, label="Model Predicted Height", color='b', linestyle='dashed')
plt.plot(data.data['Time'].to_numpy(), data.data['Height'].to_numpy(), label="Experimental Data")

plt.xlabel("Time [s]", weight='bold')
plt.ylabel("Water Height [mm]", weight='bold')
plt.title("Comparison of Model and Experimental Water Levels")
plt.legend()
plt.minorticks_on()
plt.grid(which='major', linewidth=0.5)
plt.grid(which='minor', linewidth=0.2)
plt.show()