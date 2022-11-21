import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as window

if __name__ =='__main__':
    t = np.arange(0, 100)
    plt.plot(t, window.hann(t.size), label="hann")
    plt.plot(t, window.tukey(t.size), label="tukey")
    plt.plot(t, window.blackman(t.size), label="Blackman")
    plt.plot(t, window.kaiser(t.size, 3), label="kaiser")
    plt.legend()
    plt.show()