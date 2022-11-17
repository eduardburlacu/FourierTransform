from main import generate_random_signal
fs = 4.0  # Sample and evaluate the data at this frequency for the period T
N = 5
start = 0
end = 100.0
t, x, t_show, x_show= generate_random_signal(fs,N,start,end,weight_constant=True)
#Now use 3b1b's insight into how unmixing works.
r = np.abs(x)
#omega = 2**(np.linspace(0., T * fs, 20))
omega = 0.08 * 180/ np.pi
theta = omega * t
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
ax.set_rlabel_position(-22.5)
ax.grid(True)
plt.show()