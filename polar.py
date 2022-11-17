from main import generate_random_signal, generate_sine_signal
import matplotlib.pyplot as plt
import numpy as np
fs = 10.0  # Sample and evaluate the data at this frequency for the period T
N = 5
start = 0
end = 50.0
t, x, t_show, x_show= generate_sine_signal([1.98,],fs,N,start,end,weight_constant=False)
#Now use 3b1b's insight into how unmixing works.
r = np.abs(x)
#omega = 2**(np.linspace(0., T * fs, 20))
omega = 1.98
theta = omega * t * 180/ np.pi
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
ax.set_rlabel_position(-22.5)
ax.grid(True)
plt.show()