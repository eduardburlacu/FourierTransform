import numpy as np
import matplotlib.pyplot as plt

def Hann_function(N=None, t=None):
    if t is not None: N=t.size
    f = np.arange(N)
    f = 1/2 * (1-np.cos( 2*np.pi* f /(N-1) ) )
    if t is not None: f *= t
    return f


#t=np.arange(0, 25)
#plt.plot(t, Hann_function(t.size,t))
#plt.show()