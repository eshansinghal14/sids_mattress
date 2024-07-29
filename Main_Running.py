from Heart_Rate import *
from Respiration_Rate import *
from SIDS_Detection import *
from matplotlib import pyplot as plt
# import board
# import adafruit_mlx90614
# import busio as io

final_hr = []
final_beats = [[], []]
final_breaths = [[], [], []]

sids_data = [[], [], []]

raw_values = np.array(main_processing('Human Testing Raw Data\\2.5 Week Old Mattress Data'))

# i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
# mlx = adafruit_mlx90614.MLX90614(i2c)
#
# config = {
#   "apiKey": "AIzaSyAmOi8hxXsZUZrHphWrw4HxcN6-Axi2kXM",
#   "authDomain": "sids-mattress.firebaseapp.com",
#   "projectId": "sids-mattress",
#   "storageBucket": "sids-mattress.appspot.com",
#   "messagingSenderId": "494307890551",
#   "appId": "1:494307890551:web:bcde0a8c446a904be77d72",
#   "measurementId": "G-GNV426LVKV"
# }
#
# firebase = pyrebase.initialize_app(config)
# db = firebase.database()

print('...Calibrating...')
for a in range(0, 8): # Calibrate
    # raw_values = np.array(main_processing('Human Testing Raw Data\\2.5 Week Old Mattress Data'))
    current_beats = determine_heart_rate(500, raw_values, final_beats, a)
    final_hr.append(60 * 1000 / sum(current_beats))
    final_beats[0].append(current_beats[0])
    final_beats[1].append(current_beats[1])
    current_breaths = get_respirations(1500, raw_values, final_breaths, a)
    final_breaths[0].append(len(current_breaths[0]))
    final_breaths[1].append(np.average(current_breaths[0]))
    final_breaths[2].append(np.average(current_breaths[1]))
# plt.plot(final_hr)
# plt.show()
# plt.close()
for a in range(int((len(raw_values[0]) - 30000) / 5000)):
    print(a)
    # raw_values = np.array(main_processing('Human Testing Raw Data\\2.5 Week Old Mattress Data'))
    current_beats = determine_heart_rate(5000, raw_values, final_beats, a)
    current_breaths = get_respirations(15000, raw_values, final_breaths, a)

    final_hr.append(60 * 1000 / sum(current_beats))
    final_beats[0].append(current_beats[0])
    final_beats[1].append(current_beats[1])
    final_breaths[0].append(len(current_breaths[0]) * 2)
    final_breaths[1].append(np.average(current_breaths[0]))
    final_breaths[2].append(np.average(current_breaths[1]))

    # if a > 24:
    #     sids_data, sids_score = detect_sids(a, final_breaths, final_hr, sids_data)

    print('Current HR: ')
    print(final_hr[-1])
    print('Current Respiration Rate:')
    print(len(current_breaths[0]) * 2)
    # print('Average Breath In:')
    # print(np.average(current_breaths[1]))
    # print('Average Breath Out:')
    # print(np.average(current_breaths[0]))

    # data = {
    #     "final_hr": str(round(final_hr[-1])),
    #     "average_hr": str(round(np.average(final_hr))),
    #     "final_rr": str(final_breaths[0][-1]),
    #     "inhale": str(round(final_breaths[1][-1])),
    #     "exhale": str(round(final_breaths[2][-1])),
    #     "sids_risk": 'Low'
    # }
    # db.child('tester').child().push(data)

    # file = open('C:\\Users\\ES\\StudioProjects\\sids_mattress_app\\assets\\app_data.txt', 'w')
    # app_data = str(round(final_hr[-1])) + ' ' + str(round(np.average(final_hr))) + ' ' + str(final_breaths[0][-1]) + ' ' + str(round(final_breaths[1][-1])) + ' ' + str(round(final_breaths[2][-1])) + ' ' + str(0)
    # print(app_data)
    # file.write(app_data)
    # file.close()

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize= [12,8])
ax = fig.add_subplot(111, projection='3d')
# ax = fig.gca(projection = '3d')
ax.plot_surface(np.array(final_hr), np.array(final_breaths[0]), np.array(final_breaths[2]), c=np.arange(len(final_hr)), cmap='YlOrRd', alpha=1)
ax.set_xlabel('Heart Rate')
ax.set_ylabel('Respiration Rate')
ax.set_zlabel('Frequency of Breath')
plt.clabel('Time Passed')
plt.show()
# results = pd.DataFrame(list(zip(final_beats[1], final_beats[0], final_hr, final_breaths[0], final_breaths[1], final_breaths[2])), columns=['S1', 'S2', 'HR', 'RR', 'Breath In Freq', 'Breath Out Freq'])
# results.to_csv('6 Month Old Results.csv')
file = open('C:\\Users\\ES\\StudioProjects\\sids_mattress_app\\assets\\app_data.txt', 'w')
file.write('-- -- -- -- -- --')
file.close()
# plt.close()
# plt.plot(np.arange(len(hr)), np.array(hr))
# plt.show()
