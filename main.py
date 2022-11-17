import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from window_functions import Hann_function as Hann

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

def plot_signal(t_show, x_show, x_true=None):
    plt.scatter(t_show, x_show, c='y', edgecolors='r')
    plt.plot(t_show, x_show, 'b')
    if x_true is not None: plt.plot(t_show, x_true, 'g')

def plot_frequency(arrays, fs):
    for array in arrays:
        X_show = np.abs(fft(array))
        freq = fftfreq(x_show.size, 1 / fs)
        plt.plot(freq, X_show)

def generate_sine_signal(sines, fs,N,start,end,weight_constant=False, to_plot=True):
    T = (end - start) / N
    t_show = np.arange(start, end, 1 / fs)
    t = np.arange(0, T, 1 / fs)
    x=0
    x_true=0
    if weight_constant:
        for sine in sines:
            A, PHASE = np.random.normal(1,0.25), np.random.random()
            x += A * np.sin(sine * t + PHASE )
            x_true += A * np.sin(sine * t_show + PHASE )
    else:
        for sine in sines:
            PHASE = np.random.random()
            x += np.sin(sine * t + PHASE )
            x_true += np.sin(sine * t_show + PHASE )
    x = Hann(t=x)
    x_true = Hann(t=x_true, N=x.size)
    x_show = np.array([])
    for i in range(N):
        x_show = np.append(x_show, x)

    if to_plot: return t_show, x_show, x_true
    else: return t,x, x_true


if __name__ =='__main__':
    fs = 28.0 # Sample and evaluate the data at this frequency for the period T
    N = 6
    start = 0
    end = 30.0
    resonance= 2* np.pi *np.array([2.23, 1.03,3.13])
    t_show, x_show, x_true = generate_sine_signal(resonance,fs,N,start,end,True)
    plot_signal(t_show, x_show, x_true)
    plt.grid()
    plt.show()
    plot_frequency([x_show, x_true],fs)
    plt.grid()
    plt.show()
