import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import rfftfreq, rfft


# raw_values = np.array(main_processing('Human Testing Raw Data\\Eshan Mattress Data.txt')[0:4])


def determine_heart_rate(distance_between, raw_values, final_beats, a):
    large_window_size = 10000
    window_size = 80
    dif_beats = []
    val_heights = [1, 1, 1, 8]
    current_beat = []
    for b in range(4):
        interval_values = raw_values[b][a * distance_between:a * distance_between + large_window_size]
        # interval_values = raw_values[b]
        i = window_size + 40
        beats = []
        # plt.plot(interval_values)
        # plt.show()
        # plt.close()
        while i < int(len(interval_values) - window_size):
            current_values = interval_values[i:i + window_size]
            current_values = current_values - np.average(current_values)
            # Store x and y values from FFT in dataframe
            yf = np.abs(rfft(current_values))
            xf = rfftfreq(len(current_values), 1 / 1000)

            # Cancel noise
            frequencySum = np.zeros(window_size)
            for j in range(0, 40, 10):
                profile = raw_values[b][i - j - window_size:i - j]
                profile = profile - np.average(profile)
                frequencySum += np.abs(rfft(profile))
                profile = raw_values[b][i + window_size + j:i + j + window_size * 2]
                profile = profile - np.average(profile)
                frequencySum += np.abs(rfft(profile))
            frequencySum /= 20
            cleaned_yf = np.copy(yf)
            cleaned_yf -= frequencySum

            fft_values = pd.DataFrame(list(zip(current_values, xf, cleaned_yf)), columns=['raw', 'xf', 'yf'])
            fft_values = fft_values.sort_values(by='yf',
                                                ascending=False)  # Sort values in fft_values to determine most prominent frequencies
            if (abs(fft_values['raw']) > val_heights[
                b]).sum() > 10:  # Make sure raw data has some spikes to indicate heart rate
                # Record as beat if it could be heart beat
                for j in list(range(0, 2)):
                    val = fft_values['xf'].iloc[j]
                    if 0.9 * abs(fft_values['yf'].iloc[0]) < fft_values['yf'].iloc[j] and fft_values['yf'].iloc[
                        j] > 10:
                        if abs(250 - val) < 5 or abs(175 - val) < 5:
                            beats.append(i)
                            # plt.plot(current_values)
                            # plt.show()
                            # plt.close()
                            # plt.plot(xf, yf)
                            # plt.show()
                            # plt.close()
                            # plt.plot(xf, cleaned_yf)
                            # plt.show()
                            # plt.close()
                            i += 80
                            break
            i += 20

        # Eliminate close beats
        for j in range(len(beats) - 1):
            dif_beats.append(beats[j + 1] - beats[j])

    dif_beats.sort()
    frequency = np.zeros(shape=(2, 8), dtype=int)
    # plt.hist(dif_beats, bins=8)
    # plt.show()
    # plt.close()
    frequency[0] = ((plt.hist(dif_beats, bins=8)[0]).astype(int)).tolist()
    plt.close()
    counter = 0
    for j in range(len(frequency[0])):
        sum_counter = 0
        for k in range(counter, counter + frequency[0][j]):
            sum_counter += dif_beats[k]
        if frequency[0][j] == 0:
            frequency[1][j] = 0
        else:
            frequency[1][j] = sum_counter / frequency[0][j]
        counter += frequency[0][j]

    first = frequency[1][0]
    sec = frequency[1][1]
    best_beats = np.full((3, 80), 2147483648)
    frequency = frequency[:, np.flip(frequency[0].argsort())]
    # print(frequency)
    for k in range(100, first + 100, 10):
        for l in range(100, sec + 100, 10):
            accuracy_counter = 0.0
            for j in range(5):
                num = frequency[1][j]
                remainder = num % (k + l)
                subk = abs(remainder - k)
                subl = abs(remainder - l)
                norm = abs(num - (k + l) * round(num / (k + l)))
                num_beats = 0
                if subk < norm and subk < subl:
                    remainder -= k
                    num -= k
                    num_beats += 1
                elif subl < norm and subl < subk:
                    remainder -= l
                    num -= l
                    num_beats += 1
                else:
                    remainder = norm
                num_beats += round(num * 2 / (k + l))
                accuracy_counter += ((abs(remainder) + 1) * (num_beats ** 2) * (frequency[0][j] ** 1 / 2))
            temp_best_beats = np.hstack((best_beats, np.array([[accuracy_counter], [k], [l]])))
            temp_best_beats = temp_best_beats[:, np.flip(temp_best_beats[0].argsort())]
            best_beats = np.delete(temp_best_beats, obj=0, axis=1)

    if len(final_beats[0]) < 12:
        if best_beats[1][-1] > 10000:
            current_beat = [final_beats[0][-1], final_beats[1][-1]]
        else:
            current_beat = [best_beats[1][-1], best_beats[2][-1]]
    else:
        predicted_s2 = np.average(np.array(final_beats[0])[-12:])
        predicted_s1 = np.average(np.array(final_beats[1])[-12:])
        predicted_beat = predicted_s1 + predicted_s2
        # print(predicted_s1)
        # print(predicted_s2)
        # print(best_beats)
        if best_beats[1][-1] > 10000:
            current_beat = [predicted_s2, predicted_s1]
        else:
            beat_found = False
            for j in range(len(best_beats[0]) - 1, 0, -1):
                if abs(best_beats[1][j] - predicted_s2) < .2 * predicted_s2 and abs(
                        best_beats[2][j] - predicted_s1) < .3 * predicted_s1 and 60 * 1000 / (
                        best_beats[1][j] + best_beats[2][j]) < 220:
                    current_beat = [best_beats[1][j], best_beats[2][j]]
                    beat_found = True
                    break
            # if not beat_found:
                # for j in range(len(best_beats[0]) - 1, 0, -1):
                #     if abs(best_beats[1][j] - predicted_s2) < .2 * predicted_s2 and abs(
                #             best_beats[2][j] - predicted_s1) < .3 * predicted_s1:
                #         final_beats[0].append(best_beats[1][j])
                #         final_beats[1].append(best_beats[2][j])
                #         beat_found = True
                #         break
            if not beat_found:
                # print('BEAT NOT FOUND!!!')
                # final_beats[0].append(np.nan)
                # final_beats[1].append(np.nan)
                current_beat = [predicted_s2, predicted_s1]
    # print(final_beats)
    # print(final_beats[0][-1])
    # print(final_beats[1][-1])
    # print(60 * 1000 / (final_beats[0][-1] + final_beats[1][-1]))
    return current_beat

#
# for a in range(0, 8): # Calibrate
#     determine_heart_rate(500)
# for a in range(int((len(raw_values[0]) - large_window_size) / 5000)):
#     determine_heart_rate(5000)
# hr = [60 * 1000 / sum(x) for x in zip(*final_beats)]
# print(hr)
# print(np.average(np.array(hr)[10:]))
# plt.close()
# plt.plot(np.arange(len(hr)), np.array(hr))
# plt.show()
