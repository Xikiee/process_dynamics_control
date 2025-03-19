import numpy as np
import matplotlib.pyplot as plt 
import control as ctrl

w = np.linspace(10**-1,10**5,100000)
ab = 20*np.log10(100) + 20*np.log10(np.sqrt(w**2 +1)) - 20* np.log10(np.sqrt(w**2 +100**2)) -20* np.log10(np.sqrt(w**2 +1000**2))
phase = np.degrees(np.arctan(-w) - np.arctan(w/100))


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for ab
axs[0].semilogx(w, ab, label='Magnitude (dB)')
axs[0].set_title('Magnitude Response')
axs[0].set_xlabel('Frequency (rad/s)')
axs[0].set_ylabel('Magnitude (dB)')
axs[0].grid(True)
axs[0].legend()

# Plot for phase
axs[1].semilogx(w, phase, label='Phase (radians)', color='orange')
axs[1].set_title('Phase Response')
axs[1].set_xlabel('Frequency (rad/s)')
axs[1].set_ylabel('Phase (radians)')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()



# ###### checking the answer with contorl

# num = [100,1000]
# den = [1,1100,1000*100]
# system = ctrl.TransferFunction(num,den)
# ctrl.bode_plot(system)
# plt.show()