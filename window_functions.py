import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as window

def Hann_function(N=None, t=None):
    if N is None: N=t.size
    f = np.arange(t.size)
    f = 1/2 * (1-np.cos( 2*np.pi* f /(N-1) ) )
    if t is not None: f *= t
    return f

#def Tukey_function(N=None, t=None, a=0.5):

if __name__ =='__main__':
    t=np.arange(0, 25)
    plt.plot(t, window.hann(t.size), label="hann")
    plt.plot(t, window.tukey(t.size), label="tukey")
    plt.plot(t, window.kaiser(t.size, 3), label="kaiser")
    plt.legend()
    plt.show()