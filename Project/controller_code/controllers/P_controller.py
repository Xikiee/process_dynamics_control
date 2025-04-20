import MAPort_6EX80
import numpy as np
import time

MyMAPort=MAPort_6EX80.MAPort_6EX80()

SimLength = 600 #s
time_scale = 10 #frequencey of the simulation
dt = 1/time_scale
I = 0
fv_ss = 0.059 * 1e-3 #m3/s
delta_fv = 0.059 * 1e-3 #m3/s
h_set = 0.3 #m
tc = 10 #s
e_prev = 0 #m

Kp = 5893.91 #ask Nela
tau = 90.73

# Kc = tau/Kp/tc #ziegler nichols
Kc = 47.00   
ti = tau
td = 0.1 * tc
UpperLevel = 0.4 #m, to avoid overflow
fv = 0
Normal_Operation = True #True if the system is in normal operation, False if the system is in overflow


for i in range(SimLength):
    targ_height = 350
    Height = MyMAPort.ReadHeight()
    error = targ_height - Height
    valve = MyMAPort.WriteValvePosition(0.3)
    Pumpcontrol = error * Kc
    Pumpcontrol = max(0, min(0.1, Pumpcontrol))
    MyMAPort.WritePumpFlow(Pumpcontrol)
    print(i, Pumpcontrol, Height)
    time.sleep(1)