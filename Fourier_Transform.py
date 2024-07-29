import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfftfreq, rfft


# values = []
# file = open('test.txt', 'r')
# # file = open('Heart Rate Speaker Testing/Heart Beat Testing 60 BPM.txt', 'r')
# for i in file:
#     values.append(int(i))
# file.close()

def fourier_transform(values):
    values = np.array(values)
    values = values - np.mean(values)
    print(len(values))

    # plt.plot(values)
    # plt.show()

    N = len(values)

    yf = rfft(values)
    print(yf)
    xf = rfftfreq(N, 1 / 3000)

    plt.plot(xf, np.abs(yf))
    plt.show()
