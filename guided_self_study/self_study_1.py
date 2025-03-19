import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

#problem 3 part a

def ode_solver(t,y):
    dy_dt = y[1]
    d2y_dt = -2*y[1] -2*y[0] +1
    return dy_dt, d2y_dt

#initial values 
y0 = [2,-3/2]

t_span = (0,10)
# t_eval = np.linspace(0,10,100)
sol = solve_ivp(ode_solver,t_span,y0,method = 'RK45')

plt.plot(sol.t, sol.y[0], label = 'dy_dt solution')
plt.plot(sol.t, sol.y[1], label = 'dy2_dt solution')
plt.xlabel("time (s)")
plt.ylabel('y value')
plt.legend()
plt.show()

#problem 3 part b 

def ode_solver2(t,y):
    dy_dt = 

