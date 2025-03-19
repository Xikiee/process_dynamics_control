import numpy as np
import matplotlib.pyplot as plt


#plotting a function 
def plot_function(t, title, xlabel, ylabel):
    y = 2*(1-np.exp(-t/4)) - (1-np.exp(-t))
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel( xlabel, weight='bold')
    plt.ylabel( ylabel, weight='bold')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.show()

plot_function(np.linspace(0, 10, 100), "k_1=2,k_2=1,T_1=4,T_2=1", 't', 'y(t)')

#plotting a function 
def plot_function(t, title, xlabel, ylabel):
    y = 2*(1-np.exp(-4*t)) - (1-np.exp(-t))
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel( xlabel, weight='bold')
    plt.ylabel( ylabel, weight='bold')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.show()

plot_function(np.linspace(0, 10, 100), "k_1=2,k_2=1,T_1=1/4,T_2=1", 't', 'y(t)')