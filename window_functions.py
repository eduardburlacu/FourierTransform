import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as window
from math import log as ln

if __name__ =='__main__':
    t = np.arange(0, 100)
    plt.plot(t, window.hann(t.size), label="hann")
    plt.plot(t, window.tukey(t.size), label="tukey")
    plt.plot(t, window.blackman(t.size), label="Blackman")
    plt.plot(t, window.kaiser(t.size, 3), label="kaiser")
    plt.plot(t,window.exponential(t.size,0,tau= -(t.size-1) / ln(0.008), sym=False),label='Poisson')
    plt.legend()
    plt.grid()
    plt.show()