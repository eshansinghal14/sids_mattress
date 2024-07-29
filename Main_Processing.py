import numpy as np
import binascii


def main_processing(file):
    raw_values = []
    values = [[], [], [], [], [], []]
    heart_rates = []

    file = open(file, 'rb')
    while True:  # Read file and store ASCII vals in array
        val = file.read(1)
        if not val:
            break
        # print(val)
        raw_values.append(ord(val))
    file.close()

    raw_values = np.array(raw_values[31*22000:-11000])  #2.5 Week
    # raw_values = np.array(raw_values[22000:-11000])
    # raw_values = np.array(raw_values[-180000:])
    # raw_values = np.array(raw_values)
    # Start array at 0, 0
    maxes = np.where(np.array(raw_values) == 0)[0]
    spacers = []
    for i in range(len(maxes) - 2):
        if maxes[i] + 1 == maxes[i + 1] and not maxes[i] + 2 == maxes[i + 2]:
            spacers.append(maxes[i])
    # print(spacers[10:100])
    # print(raw_values[1000:1100])
    # print(len(raw_values))
    for i in range(len(spacers) - 1):  # Converting ASCII values to raw int sensor data
        if spacers[i + 1] - spacers[i] < 22 and len(values) > 10:  # Make sure not too many
            # print(spacers[i + 1] - spacers[i])
            # print(spacers[i])
            for j in range(4):
                if j > 3:
                    values[j].append(np.array(values[j][-3:]))
                else:
                    values[j].append(values[j][-1])
        if spacers[i + 1] - spacers[i] > 21: # Make sure works even if there are 26 vals in between
            for j in range(spacers[i] + 2, spacers[i] + 22, 2):
                ascii = int(raw_values[j]).to_bytes(1, 'little') + int(raw_values[j + 1]).to_bytes(1, 'little')
                val = int.from_bytes(ascii, byteorder='little', signed=False)
                if val > 1024:
                    print(val)
                index = int((j - spacers[i]) / 2) - 1
                if index > 3:
                    index = 5 - index % 2
                # if(val > 1024):
                #     print(j)
                values[index].append(val)

    # print(values[0:1000])
    # print(values[0])
    # print(len(values[1]))
    print(len(values[1]))

    return values
