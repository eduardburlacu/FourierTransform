import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
def generate_random_signal(fs,N,start,end, weight_constant=False):
    T = (end - start) / N
    t_show = np.arange(start, end, 1 / fs)
    t = np.arange(0, T, 1 / fs)
    x = np.random.random() * np.sin(np.random.random() * t)
    points= int(T * fs)
    if weight_constant:
        for i in range(points//3):
            x += np.random.random() * np.sin(np.random.normal(1.0, 0.1) * i * t+ np.random.random())
    else:
        for i in range(points//3):
            x += np.sin(np.random.normal(1.0, 0.1) * i * t +np.random.random())
    x_show = np.array([])
    for i in range(N):
        x_show = np.append(x_show, x)
    return t,x, t_show, x_show

def plot_signal(t_show, x_show):
    plt.scatter(t_show, x_show, c='y', edgecolors='r')
    plt.plot(t_show, x_show)
    plt.grid()
    plt.show()

def generate_sine_signal(sines, fs,N,start,end, weight_constant=False):
    T = (end - start) / N
    t_show = np.arange(start, end, 1 / fs)
    t = np.arange(0, T, 1 / fs)
    x=0
    if weight_constant:
        for sine in sines:
            x += np.random.normal(1,0.25) * np.sin(sine * t + np.random.random())
    else:
        for sine in sines:
            x += np.sin(sine * t + np.random.random())
    #x -= min(x)
    x_show = np.array([])
    for i in range(N):
        x_show = np.append(x_show, x)
    return t,x, t_show, x_show


if __name__ =='__main__':
    fs = 10.0 # Sample and evaluate the data at this frequency for the period T
    N = 5
    start = 0
    end = 50.0
    resonance= 2* np.pi *np.array([2., 1.0])
    t, x, t_show, x_show = generate_sine_signal(resonance,fs,N,start,end,True)
    plot_signal(t_show, x_show)
    X_show = np.abs(fft(x_show))
    freq = fftfreq(x_show.size, 1/fs)
    plot_signal(freq, X_show)

