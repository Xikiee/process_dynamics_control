import numpy as np
import matplotlib.pyplot as plt 

data = r"Project\data\process_data\experiment_6.ChannelGroup_0__CGcomment_xmlns=_http___www.asam.net_mdf_v4___TX___TX___CGcomment_.csv"

def extraction(data):
    data = np.genfromtxt(data, delimiter= ',', skip_header=1)
    timestamps = data[:,0]
    height_t1 = data[:,1]
    pump_left = data[:,2]
    valve_pos_ld = data[:,3]
    
    combined_array = np.column_stack((timestamps, height_t1, pump_left, valve_pos_ld))
    return combined_array

result = extraction(data)

# print(result)

plt.plot(result[:,0], result[:,1])
plt.xlabel('time (s)')
plt.ylabel('height (mm)')
plt.minorticks_on()
plt.grid(which = "both", linewidth = 0.5)
plt.show()