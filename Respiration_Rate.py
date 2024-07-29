import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import rfftfreq, rfft
from scipy.signal import find_peaks
import scipy.signal as signal
import noisereduce as nr
from Fourier_Transform import *
from Main_Processing import *

# raw_values = np.array(main_processing('Human Testing Raw Data\\Akul Mattress Data.txt')[4:6])


def get_respirations(distance_between, raw_values, final_breaths, a):
    large_window_size = 90000
    window_size = 600
    noise = raw_values[4][a * distance_between:a * distance_between + large_window_size]
    actual = raw_values[5][a * distance_between:a * distance_between + large_window_size]
    # noise = raw_values[4]
    # actual = raw_values[5]
    # plt.plot(actual)
    # plt.show()
    # plt.close()
    reduced_noise = nr.reduce_noise(y=actual, sr=3000, y_noise=noise, time_mask_smooth_ms=86, prop_decrease=.8)
    # plt.plot(reduced_noise)
    # plt.show()
    # plt.close()
    i = 0
    breath_freqs = []
    while i < int(len(reduced_noise) - window_size):
        current_values = reduced_noise[i:i + window_size]
        # plt.plot(current_values)
        # plt.show()
        # plt.close()
        current_values = current_values - np.average(current_values)
        yf = np.abs(rfft(current_values))[120:600]
        xf = np.array(rfftfreq(len(current_values), 1 / 3000))[120:600]
        # plt.plot(np.array(rfftfreq(len(current_values), 1 / 3000)), np.abs(rfft(current_values)))
        # plt.show()
        # plt.close()
        # plt.plot(xf, yf)
        # plt.show()
        # plt.close()
        breath_freqs.append(np.average(xf, weights=yf))
        i += 60

    # plt.plot(breath_freqs)
    # plt.show()
    # plt.close()
    breath_freqs = signal.savgol_filter(breath_freqs, 71, 10)
    # plt.plot(breath_freqs)
    # plt.show()
    # plt.close()
    q3 = np.percentile(breath_freqs, 75, interpolation='midpoint')
    q1 = np.percentile(breath_freqs, 25, interpolation='midpoint')
    breath_peaks = find_peaks(breath_freqs, height=breath_freqs.mean() + .5 * (q3 - q1),
                              prominence=.5 * (q3 - q1))[0]
    print(breath_peaks)
    breaths = [[], []]
    for i in range(len(breath_peaks)-1):
        breaths[0].append(breath_freqs[int(breath_peaks[i])])
        breaths[1].append(breath_freqs[int((breath_peaks[i] + breath_peaks[i+1]) / 2)])
    return breaths

# for a in range(int((len(raw_values[0]) - large_window_size) / distance_between)):
#     current_breaths = get_respirations()
