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

Kc = tau/Kp/tc #ziegler nichols
ti = tau
td = 0.1 * tc
UpperLevel = 0.4 #m, to avoid overflow
fv = 0
Normal_Operation = True #True if the system is in normal operation, False if the system is in overflow
for i in range(SimLength*time_scale):
    Height = MyMAPort.ReadHeight()/1000
    if Height > UpperLevel:
        Normal_Operation = False
    
    if Normal_Operation:
        valve_pos = 0.3
        MyMAPort.WriteValvePosition(valve_pos)
        e = h_set - Height
        P = Kc*e
        I += e*dt
        D = (e-e_prev)/dt
        e_prev = e
        delta_fv = Kc*e + (Kc/ti)*I + Kc*td*D #to make it just P control

        fv = fv_ss + delta_fv
        fv = fv*1000 #convert to l/s
        if fv < 0:
            fv = 0
            I -= e*dt #integral windup
        if fv > 1e-1:
            fv = 1e-1
            I -= e*dt #integral windup
        MyMAPort.WritePumpFlow(fv)
        print(e, ' ', fv)
    else:
        MyMAPort.WritePumpFlow(0.0)
    time.sleep(dt)
