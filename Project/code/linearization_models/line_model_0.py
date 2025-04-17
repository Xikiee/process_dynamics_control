import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def linearized_model(time,p,f,h0, a3):
    A1 = 0.0154  # Area of tank in m^2
    A2 = 4.91e-4  # Area of drain pipe in m^2
    k_pump = 1.1  
    g = 9.81  
    k_opening = 5  # Fixed within range 2 to 8
    h0  # Initial height in the tank
    
    height = np.zeros_like(time)
    height[0] = h0
    dt = time[1] - time[0]
    for i in range(0, len(time) - 1):

        inflow = (k_pump/A1) * p 
        outflow = (k_opening * f * (A2/A1) + (k_opening * f * (a3/A1))) * np.sqrt(2*g*height[i])

        #0, 0+ dh
        height[i+1] = height [i] + (inflow - outflow)*dt                                                                  #i+1 so that the initial height remains as 0
    
    return time, height





t = np.linspace(0,100,10000)
p = 0.06
v = 0.3
time, height = linearized_model(t,p,v, 323.729770, 0.00015 )


plt.plot(time, height)
plt.show()
#### i need to make sure that the first height is actually h0, after that i add/subtract the height





# import numpy as np
# import matplotlib.pyplot as plt

# def linearized_model(time, p, f, h0, a3):
#     A1 = 0.0154  # Area of tank in m^2
#     A2 = 4.91e-4  # Area of drain pipe in m^2
#     k_pump = 1.1  
#     g = 9.81  
#     k_opening = 5  # Fixed within range 2 to 8

#     height = np.zeros_like(time)
#     height[0] = h0

#     for i in range(0, len(time) - 1):
#         inflow = (k_pump / A1) * p
#         outflow = k_opening * f * ((A2 + a3) / A1) * np.sqrt(2 * g * height[i])
#         dt = time[i+1] - time[i]
#         height[i+1] = height[i] + (inflow - outflow) * dt

#     return time, height

# # Example usage
# t = np.linspace(0, 100, 10000)
# p = 0.06
# f = 0.3
# time, height = linearized_model(t, p, f, h0=0, a3=0.00015)

# plt.plot(time, height)
# plt.xlabel('Time (s)')
# plt.ylabel('Height (m)')
# plt.title('Tank Height Over Time')
# plt.grid()
# plt.show()
