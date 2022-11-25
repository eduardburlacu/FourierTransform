import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import scipy.signal.windows as window

def plot_signal(t_show, x_show, x_true=None, z=None):
    if x_true is not None: plt.plot(t_show, x_true, 'g', label="Sampled Signal")    # Plot the sampled signal
    if z is not None: plt.plot(t_show, z, '-.' 'r', label='Original Waveform')      # Plot the original signal to be measured
    plt.plot(t_show, x_show, '--' 'b', label="New Signal")
    x_show = np.tile(window.tukey(int(x_show.size / N)), N)                         # Plot the corrected signal
    plt.plot(t_show, x_show, 'orange', label="Window Function")                     # Plot the window function used
    plt.legend()
    plt.title("Tukey")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def plot_frequency(arrays, fs):
    for array in arrays:
        X_show = np.abs(fft(array))                                                 # Perform fast fourier transform on array
        freq = fftfreq(X_show.size, 1 / fs)                                         # Find corresponding frequencies
        plt.plot(freq, X_show)
    plt.title("Tukey")                                                              # Plot frequency spectrum
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def generate_signal(sines, fs, N, start, end, weight_constant=False, to_plot=True):
    t_show = np.arange(start, end, 1 / fs)                                          # Create array of time intervals for longer period
    t = np.arange(0, (end - start) / N, 1 / fs)                                     # Split sampled period into time intervals
    x = 0                                                                           # Initialize arrays
    x_true = 0
    z=0
    if weight_constant:                                                             # If varying amplitudes are desired
        for sine in sines:
            A, PHASE = np.random.normal(1,0.25), 2 * np.pi * np.random.random()     # Generate random amplitude and phase
            x += A * np.sin(sine * t + PHASE)                                       # Add array of values for sampling period
            z += A * np.sin(sine * t_show + PHASE)                                  # Add array of values for longer period
    else:                                                                           # If constant amplitude is desired
        for sine in sines:
            PHASE = 2 * np.pi * np.random.random()                                  # Generate random phase
            x += np.sin(sine * t + PHASE)                                           # Add array of values for sampling period
            z += np.sin(sine * t_show + PHASE)                                      # Add array of values for longer period
    x_true = np.tile(x, N)                                                          # Original randomized signal sampled for period T
    x = np.multiply(x, window.tukey(x.size, 3))                                     # Original signal multiplied by window function
    x_show = np.tile(x, N)
    if to_plot: return t_show, x_show, x_true, z
    else: return t, x, x_true, z

if __name__ =='__main__':
    fs = 28.0                                                                       # Sample and evaluate the data at this frequency for the period T
    N = 6                                                                           # Number of repetitions of sample
    start = 0                                                                       # Start time
    end = 30.0                                                                      # End time
    resonance= 2 * np.pi * np.array([2.23, 1.03, 3.13])                             # Resonance frequencies of structure
    t_show, x_show, x_true, z = generate_signal(resonance, fs, N, start, end, True) # Generate random signal made up of resonance frequencies with random phases and amplitudes
    plot_signal(t_show, x_show, x_true)                                             # Plot random, corrected and window function signals
    plot_frequency([x_true, x_show, z], fs)                                         # Plot frequency spectra
