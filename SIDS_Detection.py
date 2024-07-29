import numpy as np
from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, GRU, Dense
import random
from sklearn import metrics


def detect_sids(time, final_breaths, final_hr, sids_data):
    input_data = []
    # for i in range(24, time):
    #     rr_difs = np.subtract(final_breaths[0][-24:-1], np.average(final_breaths[0]))
    #     hr_difs = np.subtract(final_hr[-24:-1], np.average(final_hr))
    #     bo_difs = np.subtract(final_breaths[1][-24:-1], np.average(final_breaths[1]))
    #     bi_difs = np.subtract(final_breaths[2][-24:-1], np.average(final_breaths[2]))
    #     input_data.append(np.concatenate(([time], rr_difs, hr_difs, bo_difs, bi_difs), axis=None))
    # print(input_data)
    # model = OneClassSVM(gamma='scale', nu=.01)
    # model.fit(input_data)
    # print(model.get_params())
    #
    # score = model.predict(input_data)[-1]
    #
    # if score == 1:
    #     sids_data[0].append(np.nan)
    #     sids_data[1].append(np.nan)
    #     sids_data[2].append(np.nan)
    # else:
    #     sids_data[0].append(final_breaths[0][-1])
    #     sids_data[1].append(np.average(final_breaths[1][-1], final_breaths[2][-1]))
    #     sids_data[2].append(final_hr[-1])

    # brady_index = -1
    # for i in range(time-1):
    #     if sids_data[2][i] > 60 and sids_data[2][i+1] < 60:
    #         brady_index = i+1
    #
    sids_score = 0
    # for i in range(brady_index, brady_index + 12 * 30):  # HR below 15
    #     if sids_data[2][i] < 15:
    #         sids_score += 1 / abs(12 * 7.5 - i + brady_index)
    #         break
    # for i in range(brady_index - 6, brady_index + 15 * 12):  # Apnea started
    #     if sids_data[0][i] < 3:
    #         apnea_time = 0
    #         while True:
    #             apnea_time += 1
    #             if sids_data[0][i + apnea_time] > 3:
    #                 break
    #         sids_score += apnea_time / abs(12 * 2.7 - i + brady_index)
    #         break
    # for i in range(brady_index - 11 * 12, brady_index + 15 * 12):
    #     if sids_data[1][i] > 1.1 * np.average(final_breaths[1], final_breaths[2]):
    #         gasping_time = 0
    #         while True:
    #             gasping_time += 1
    #             if sids_data[1][i + gasping_time] < 1.1 * np.average(final_breaths[1], final_breaths[2]):
    #                 break
    #         sids_score += gasping_time / abs(i - brady_index)
    #         break

    # inputs = tf.random.normal([32, 10, 8])
    # gru = tf.keras.layers.GRU(4)

    return sids_data, sids_score


def occ_tester():
    input_data = []
    # hr_vals = []
    # true_pos = 0
    # false_pos = 0
    roc_points = [[], []]
    nu_vals = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 70, 100, 200, 300, 400, 1000]
    # nu_vals = [300]
    for n in nu_vals:
        print(n)
        model = OneClassSVM(gamma='scale', nu=n/1000)
        for i in range(0, 10):
            for x in range(0, 1440):
                hr = 0.00007716049382716 * x * x - 0.11111111111111 * x + 120 + random.randrange(-5, 5)
                rr = 0.00003858024691358 * x * x - 0.055555555555556 * x + 60 + random.randrange(-2, 2)
                gasp = 0
                if random.randrange(0, 1440) < 3:
                    gasp = 1

                input_data.append([x, hr, rr, gasp])
                # print(m)
                # hr_vals.append(gasp)
        # print(input_data)
        model.fit(input_data)

        input_data = []
        # vals = []
        for x in range(0, 720):
            hr = 0.00007716049382716 * x * x - 0.11111111111111 * x + 120 + random.randrange(-5, 5)
            rr = 0.00003858024691358 * x * x - 0.055555555555556 * x + 60 + random.randrange(-2, 2)
            gasp = 0
            if random.randrange(0, 1440) < 3:
                gasp = 1

            input_data.append([x, hr, rr, gasp])
            # vals.append(gasp)
            # if model.predict([x, hr, rr, gasp])[-1] == 1:
            #     true_pos += 1

        # print(model.predict(input_data))
        true_neg = np.count_nonzero(model.predict(input_data) == 1)

        hr_fake = [80, 87, 92, 88, 100, 113, 107, 120, 117, 119, 100, 93, 84, 78, 71, 67, 60, 54, 50, 47]
        rr_fake = [30, 15, 0, 0, 0, 7, 0, 6, 13, 21, 37, 32, 40, 35, 37, 39, 41, 38, 36, 34]
        gasp_fake = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

        input_data = []
        for x in range(20):
            input_data.append([x + 720, hr_fake[x], rr_fake[x], gasp_fake[x]])
            # vals.append(gasp_fake[x])
            # if model.predict([x + 720, hr_fake[x], rr_fake[x], gasp_fake[x]])[-1] == 1:
            #     false_pos += 1
        true_pos = np.count_nonzero(model.predict(input_data) == -1)

        roc_points[0].append(1 - true_neg / 720)
        roc_points[1].append(true_pos / 20)

        print(true_pos)
        print(true_neg)


        # print(true_pos)
        # print(false_pos)
        # plt.plot(vals)
        # plt.show()

    plt.close()
    print(roc_points)
    ax = plt.gca()
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    plt.plot(roc_points[0][1:], roc_points[1][1:])
    plt.show()
    # roc_points = np.array(roc_points)
    # roc_points = roc_points[:, np.flip(roc_points[0].argsort())]
    print(metrics.auc(roc_points[0][2:], roc_points[1][2:]))


def rnn_tester():
    patient = -1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(3):
        patient += 1
        print(patient)
        hr = []
        apneas = [0]*48
        gasps = [0]*48
        sids_time = [0]*48
        counter = 0
        file = open('RNN Memory Recording Data.txt', 'rb')
        for line in file:
            counter += 1
            if int(counter / 7) == patient:
                for num in line.split():
                    if counter % 7 == 5:
                        apneas[int(num)-1] = 1
                    elif counter % 7 == 6:
                        gasps[int(num)-1] = 1
                    elif counter % 7 == 0:
                        for a in range(48):
                            if a > int(num)-36:
                                sids_time[a] = 1
                    else:
                        hr.append(int(num)/250)
        # plt.plot(apneas)
        # plt.show()

        total_data = []
        for j in range(len(hr)):
            total_data.append([hr[j], apneas[j], gasps[j]])
        # print(total_data)
        if i == 0:
            # x_test.append(total_data)
            # y_test.append(sids_time)

            normal_data = []
            for a in range(48):
                normal_data.append([(120 + random.randrange(-10, 10)) / 250, 0, 0])
            x_test.append(normal_data)
            y_test.append([0] * 48)
        else:
            x_train.append(total_data)
            y_train.append(sids_time)
            # y_train.append(list(range(48, 0, -1)))
        print(x_train)

    for i in range(2):
        normal_data = []
        for a in range(48):
            normal_data.append([(120 + random.randrange(-10, 10)) / 250, 0, 0])
        x_train.append(normal_data)
        y_train.append([0] * 48)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(x_train.shape[1:])

    model = Sequential()
    model.add(GRU(2, input_shape=(x_train.shape[1:]), activation='tanh', return_sequences=True))
    model.add(Dropout(0))
    # model.add(GRU(2, activation='tanh'))
    # model.add(Dropout(0))
    model.add(Dense(49, activation='softmax'))
    # model.add(Dense(49, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

    print(history.history)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    plt.close()
    plt.plot(np.arange(0, 100), loss, 'r', np.arange(0, 100), val_loss, 'b')
    plt.show()

    plt.close()
    plt.plot(np.arange(0, 100), accuracy, 'r',  np.arange(0, 100), val_accuracy, 'b')
    plt.show()

# occ_tester()
# rnn_tester()
