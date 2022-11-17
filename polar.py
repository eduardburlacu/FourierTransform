from main import generate_random_signal, generate_sine_signal
import matplotlib.pyplot as plt
import numpy as np

fs = 10.0  # Sample and evaluate the data at this frequency for the period T
N = 5
start = 0
end = 50.0
resonance = 2 * np.pi * np.array([1.9, 2.7])
t, x, t_show, x_show = generate_sine_signal(resonance, fs, N, start, end)
#Now use 3b1b's insight into how unmixing works.
r = np.abs(x)
#omega = 2**(np.linspace(0., T * fs, 20))
omega = 2 * np.pi * 1.9
theta = omega * t
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, x)
ax.set_rlabel_position(-22.5)
ax.grid(True)
plt.show()