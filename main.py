import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import scipy.signal.windows as window
from scipy.io import loadmat
import os

# Function used to generate completely random signal without resonance frequencies
"""
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
    x_show = np.tile(x, N)
    return t,x, t_show, x_show
"""

def plot_signal(t_show, x_show, x_true=None):
    if x_true is not None: plt.plot(t_show, x_true, 'g', label="Signal")            # Plot the sampled signal
    plt.plot(t_show, x_show, '--' 'b', label="New Signal")                          # Plot the corrected signal
    x_show = np.tile(window.blackman(int(x_show.size/N)), N)
    plt.plot(t_show, x_show, 'orange', label="Window Function")                     # Plot the window function used
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def plot_frequency(arrays, fs):
    for array in arrays:
        X_show = np.abs(fft(array))                                                 # Perform fast fourier transform on array
        freq = fftfreq(x_show.size, 1 / fs)                                         # Find corresponding frequencies
        plt.plot(freq, X_show)                                                      # Plot frequency spectrum
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def generate_signal(sines, fs, N, start, end, weight_constant=False, to_plot=True):
    t_show = np.arange(start, end, 1 / fs)                                          # Create array of x axis values
    t = np.arange(0, (end - start) / N, 1 / fs)                                     # Split sampled period into time intervals
    x = 0                                                                           # Initialize arrays
    x_true = 0
    if weight_constant:                                                             # If varying amplitudes are desired
        for sine in sines:
            A, PHASE = np.random.normal(1,0.25), 2 * np.pi * np.random.random()     # Generate random amplitude and phase
            x += A * np.sin(sine * t + PHASE)                                       # Add array of values for each resonance frequency
    else:                                                                           # If constant amplitude is desired
        for sine in sines:
            PHASE = 2 * np.pi * np.random.random()                                  # Generate random phase
            x += np.sin(sine * t + PHASE)                                           # Add array of values for each resonance frequency
    x_true = np.tile(x, N)                                                          # Original randomized signal sampled for period T
    x = np.multiply(x, window.blackman(x.size))                                     # Original signal multiplied by window function
    x_show = np.tile(x, N)
    if to_plot: return t_show, x_show, x_true
    else: return t, x, x_true

def acquire_data():
    '''
    Imports data from the CollectedDataAugust2020 folder and returns it as Pandas DataFrame
    '''
    output=[]
    dir_name = 'CollectedDataAugust2020'
    directory = os.fsencode(dir_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mat"):
            a = loadmat(os.path.join(dir_name, filename))
            output.append((filename, pd.DataFrame(a)))
    return output

if __name__ =='__main__':
    fs = 28.0                                                                       # Sample and evaluate the data at this frequency for the period T
    N = 6                                                                           # Number of repetitions of sample
    start = 0                                                                       # Start time
    end = 30.0                                                                      # End time
    resonance= 2 * np.pi * np.array([2.23, 1.03, 3.13])                             # Resonance frequencies of structure
    t_show, x_show, x_true = generate_signal(resonance, fs, N, start, end, True)    # Generate random signal made up of resonance frequencies with random phases and amplitudes
    plot_signal(t_show, x_show, x_true)                                             # Plot random, corrected and window function signals
    plot_frequency([x_true, x_show], fs)                                            # Plot frequency spectrum