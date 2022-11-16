import numpy as np
import matplotlib.pyplot as plt
fs = 4.0 # Sample and evaluate the data at this frequency for the period T
N = 5
start = 0
end = 100.0
T = (end - start) / N
t_show = np.arange(start, end, 1/fs)
t = np.arange(0, T, 1/fs)
current = np.random.random() * np.sin(np.random.random() * t)
for i in range(int(T*fs)):
    current += np.random.random() * np.sin(np.random.random() * 2 ** i  + np.random.random() * t)
y=np.array([])
for i in range(N):
    y = np.append(y, current)

plt.scatter(t_show,y, c='y',edgecolors='r')
plt.plot(t_show,y)
plt.grid()
plt.show()