import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Define the transfer functions
G = ctrl.TransferFunction([1], [1, 1])
H = ctrl.TransferFunction([1], [1, 1, 1])

a_values = [2, -2]

# Time vector
t = np.linspace(0, 10, 1000)

# Plot for G and G1
plt.figure()
for a in a_values:
    G1 = ctrl.TransferFunction([1, a], [1, 1])
    t, y_G = ctrl.step_response(G, T=t)
    t, y_G1 = ctrl.step_response(G1, T=t)
    plt.plot(t, y_G, label=f'G(s)')
    plt.plot(t, y_G1, label=f'G1(s) with a={a}')
plt.title('Step Response of G and G1')
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.grid()

# Plot for H and H1
plt.figure()
for a in a_values:
    H1 = ctrl.TransferFunction([1, a], [1, 1, 1])
    t, y_H = ctrl.step_response(H, T=t)
    t, y_H1 = ctrl.step_response(H1, T=t)
    plt.plot(t, y_H, label=f'H(s)')
    plt.plot(t, y_H1, label=f'H1(s) with a={a}')
plt.title('Step Response of H and H1')
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.grid()

plt.show()