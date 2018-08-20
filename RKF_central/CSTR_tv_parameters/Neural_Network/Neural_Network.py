from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers, regularizers
from keras.models import model_from_json

import scipy.io as sio
import numpy as NP
from pdb import set_trace as bp
import random
import pylab as P

from casadi import *

def shuffleMonths(data):
    data = data.squeeze()
    mon = 6*24*NP.array([31,28,31,30,31,30,31,31,30,31,30,31])
    month = 6*24*NP.array([31, 59, 90, 120, 151, 181, 212 , 243, 273, 304, 334, 365])
    years = int(len(data)/(6*24*365))

    jan = NP.zeros((mon[0], years))
    feb = NP.zeros((mon[1], years))
    mar = NP.zeros((mon[2], years))
    apr = NP.zeros((mon[3], years))
    may = NP.zeros((mon[4], years))
    jun = NP.zeros((mon[5], years))
    jul = NP.zeros((mon[6], years))
    aug = NP.zeros((mon[7], years))
    sep = NP.zeros((mon[8], years))
    okt = NP.zeros((mon[9], years))
    nov = NP.zeros((mon[10], years))
    dez = NP.zeros((mon[11], years))

    for i in range(0,len(data),6*24*365):
        ind = int(i/(6*24*365))
        jan[:,ind] = data[i:i+month[0]]
        feb[:,ind] = data[i+month[0]:i+month[1]]
        mar[:,ind] = data[i+month[1]:i+month[2]]
        apr[:,ind] = data[i+month[2]:i+month[3]]
        may[:,ind] = data[i+month[3]:i+month[4]]
        jun[:,ind] = data[i+month[4]:i+month[5]]
        jul[:,ind] = data[i+month[5]:i+month[6]]
        aug[:,ind] = data[i+month[6]:i+month[7]]
        sep[:,ind] = data[i+month[7]:i+month[8]]
        okt[:,ind] = data[i+month[8]:i+month[9]]
        nov[:,ind] = data[i+month[9]:i+month[10]]
        dez[:,ind] = data[i+month[10]:i+month[11]]

    NP.random.seed(1)
    NP.random.shuffle(jan.transpose())

    NP.random.seed(2)
    NP.random.shuffle(feb.transpose())

    NP.random.seed(3)
    NP.random.shuffle(mar.transpose())

    NP.random.seed(4)
    NP.random.shuffle(apr.transpose())

    NP.random.seed(5)
    NP.random.shuffle(may.transpose())

    NP.random.seed(6)
    NP.random.shuffle(jun.transpose())

    NP.random.seed(7)
    NP.random.shuffle(jul.transpose())

    NP.random.seed(8)
    NP.random.shuffle(aug.transpose())

    NP.random.seed(9)
    NP.random.shuffle(sep.transpose())

    NP.random.seed(10)
    NP.random.shuffle(okt.transpose())

    NP.random.seed(11)
    NP.random.shuffle(nov.transpose())

    NP.random.seed(12)
    NP.random.shuffle(dez.transpose())

    tmp = NP.zeros((6*24*365,years))

    for i in range(0,years):
        tmp[0:month[0],i] = jan[:,i]
        tmp[month[0]:month[1],i] = feb[:,i]
        tmp[month[1]:month[2],i] = mar[:,i]
        tmp[month[2]:month[3],i] = apr[:,i]
        tmp[month[3]:month[4],i] = may[:,i]
        tmp[month[4]:month[5],i] = jun[:,i]
        tmp[month[5]:month[6],i] = jul[:,i]
        tmp[month[6]:month[7],i] = aug[:,i]
        tmp[month[7]:month[8],i] = sep[:,i]
        tmp[month[8]:month[9],i] = okt[:,i]
        tmp[month[9]:month[10],i] = nov[:,i]
        tmp[month[10]:month[11],i] = dez[:,i]
    return tmp.transpose().reshape(tmp.shape[0]*tmp.shape[1],1)


"""
----------------------------------------
Training Data Prep
----------------------------------------
"""

data = NP.load('60.npy')
# SchedVal = shuffleMonths(data[0,:])
# HeatRate = shuffleMonths(data[1,:])
# u_AHU1_noERC = shuffleMonths(data[2,:])
# ZoneTemp = shuffleMonths(data[3,:])
# OutdoorTemp = shuffleMonths(data[4,:])
# v_IG_Offices = shuffleMonths(data[5,:])
# v_solGlobFac_E = shuffleMonths(data[6,:])
# v_solGlobFac_N = shuffleMonths(data[7,:])
# v_solGlobFac_S = shuffleMonths(data[8,:])
# v_solGlobFac_W = shuffleMonths(data[9,:])
# u_blinds_E = shuffleMonths(data[10,:])
# u_blinds_N = shuffleMonths(data[11,:])
# u_blinds_S = shuffleMonths(data[12,:])
# u_blinds_W = shuffleMonths(data[13,:])

# NP.save('disturbances.npy', [data[4,0:6*24*365], data[3,0:6*24*365], data[5,0:6*24*365], data[6,0:6*24*365], data[7,0:6*24*365], data[8,0:6*24*365]])
load = 0
save = 1

if load == 1:
    lbub = NP.load('lbub.npy')
    x_lb = lbub[0]
    x_ub = lbub[1]
    y_lb = lbub[2]
    y_ub = lbub[3]

trainsize = 1

# retrieve simulated variables from FMU.py
# create training & test data
x_test = []
y_test = []
x_train = []
y_train = []

# not all zones have windows or radiators
zones_Heating = ['Coworking', 'Corridor', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth', 'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomM', 'RestroomW', 'Space01', 'Stairway']
zones = ['Coworking', 'Corridor','Elevator', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth', 'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomM' ,'RestroomW', 'Space01', 'Stairway']
zones_ahu = ['Coworking', 'Corridor', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth', 'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomW', 'Space01', 'Stairway']

ind = 2*len(zones_Heating) + len(zones_ahu)
for i in range(len(zones)):
    tmp = 'T_' + zones[i]
    tmp1 = 'T_' + zones[i] + '_train'
    tmp2 = 'T_' + zones[i] + '_test'
    vars()[tmp] = data[i + ind,:]
    vars()[tmp1] = vars()[tmp][0:int(trainsize*data.shape[1])]
    vars()[tmp2] = vars()[tmp][int(0.8*data.shape[1]):data.shape[1]]

    x_train.append(vars()[tmp1])
    x_test.append(vars()[tmp2])

    y_train.append(vars()[tmp1])
    y_test.append(vars()[tmp2])

ind = len(zones_Heating)
for i in range(len(zones_Heating)):
    tmp = 'SchedVal_' + zones_Heating[i]
    tmp1 = 'SchedVal_' + zones_Heating[i] + '_train'
    tmp2 = 'SchedVal_' + zones_Heating[i] + '_test'
    vars()[tmp] = data[i,:]
    vars()[tmp1] = vars()[tmp][0:int(trainsize*data.shape[1])]
    vars()[tmp2] = vars()[tmp][int(0.8*data.shape[1]):data.shape[1]]

    x_train.append(vars()[tmp1])
    x_test.append(vars()[tmp2])


    tmp = 'Heatrate_'+ zones_Heating[i]
    tmp1 = 'Heatrate_' + zones_Heating[i] + '_train'
    tmp2 = 'Heatrate_' + zones_Heating[i] + '_test'
    vars()[tmp] = data[i + ind,:]/100
    vars()[tmp1] = vars()[tmp][0:int(trainsize*data.shape[1])]
    vars()[tmp2] = vars()[tmp][int(0.8*data.shape[1]):data.shape[1]]

    y_train.append(vars()[tmp1])
    y_test.append(vars()[tmp2])


ind = 2*len(zones_Heating)
for i in range(len(zones_ahu)):
    tmp = 'u_AHU_' + zones_ahu[i]
    tmp1 = 'u_AHU_' + zones_ahu[i] + '_train'
    tmp2 = 'u_AHU_' + zones_ahu[i] + '_test'
    vars()[tmp] = data[i + ind,:]
    vars()[tmp1] = vars()[tmp][0:int(trainsize*data.shape[1])]
    vars()[tmp2] = vars()[tmp][int(0.8*data.shape[1]):data.shape[1]]

    x_train.append(vars()[tmp1])
    x_test.append(vars()[tmp2])

ind = 2*len(zones_Heating) + len(zones_ahu) + 2*len(zones) + 1

u_blinds_E = data[ind + 4]
u_blinds_N = data[ind + 5]
u_blinds_S = data[ind + 6]
u_blinds_W = data[ind + 7]

u_blinds_E_train = u_blinds_E[0:int(trainsize*data.shape[1])]
u_blinds_N_train = u_blinds_N[0:int(trainsize*data.shape[1])]
u_blinds_S_train = u_blinds_S[0:int(trainsize*data.shape[1])]
u_blinds_W_train = u_blinds_W[0:int(trainsize*data.shape[1])]

u_blinds_E_test = u_blinds_E[int(0.8*data.shape[1]):data.shape[1]]
u_blinds_N_test = u_blinds_N[int(0.8*data.shape[1]):data.shape[1]]
u_blinds_S_test = u_blinds_S[int(0.8*data.shape[1]):data.shape[1]]
u_blinds_W_test = u_blinds_W[int(0.8*data.shape[1]):data.shape[1]]

x_train.extend((u_blinds_E_train, u_blinds_N_train, u_blinds_S_train, u_blinds_W_train))
x_test.extend((u_blinds_E_test, u_blinds_N_test, u_blinds_S_test, u_blinds_W_test))



ind = 2*len(zones_Heating) + len(zones_ahu) + len(zones) + 1
for i in range(len(zones)):
    tmp = 'v_IG_' + zones[i]
    tmp1 = 'v_IG_' + zones[i] + '_train'
    tmp2 = 'v_IG_' + zones[i] + '_test'
    vars()[tmp] = data[i + ind, :]
    vars()[tmp1] = vars()[tmp][0:int(trainsize*data.shape[1])]
    vars()[tmp2] = vars()[tmp][int(0.8*data.shape[1]):data.shape[1]]

    x_train.append(vars()[tmp1])
    x_test.append(vars()[tmp2])

v_Tamb = data[2*len(zones_Heating) + len(zones_ahu) + len(zones),:]
v_Tamb_train = v_Tamb[0:int(trainsize*data.shape[1])]
v_Tamb_test = v_Tamb[int(0.8*data.shape[1]):data.shape[1]]

x_train.append(v_Tamb_train)
x_test.append(v_Tamb_test)


ind = 2*len(zones_Heating) + len(zones_ahu) + 2*len(zones) + 1
v_solGlobFac_E = data[ind]
v_solGlobFac_N = data[ind + 1]
v_solGlobFac_S = data[ind + 2]
v_solGlobFac_W = data[ind + 3]

v_solGlobFac_E_train = v_solGlobFac_E[0:int(trainsize*data.shape[1])]
v_solGlobFac_N_train = v_solGlobFac_N[0:int(trainsize*data.shape[1])]
v_solGlobFac_S_train = v_solGlobFac_S[0:int(trainsize*data.shape[1])]
v_solGlobFac_W_train = v_solGlobFac_W[0:int(trainsize*data.shape[1])]

v_solGlobFac_E_test = v_solGlobFac_E[int(0.8*data.shape[1]):data.shape[1]]
v_solGlobFac_N_test = v_solGlobFac_N[int(0.8*data.shape[1]):data.shape[1]]
v_solGlobFac_S_test = v_solGlobFac_S[int(0.8*data.shape[1]):data.shape[1]]
v_solGlobFac_W_test = v_solGlobFac_W[int(0.8*data.shape[1]):data.shape[1]]

x_train.extend((v_solGlobFac_E_train, v_solGlobFac_N_train, v_solGlobFac_S_train, v_solGlobFac_W_train))
x_test.extend((v_solGlobFac_E_test, v_solGlobFac_N_test, v_solGlobFac_S_test, v_solGlobFac_W_test))


v_windspeed = data[ind + 8]
v_windspeed_train = v_windspeed[0:int(trainsize*data.shape[1])]
v_windspeed_test = v_windspeed[int(0.8*data.shape[1]):data.shape[1]]

Hum_amb = data[ind + 9]
Hum_amb_train = Hum_amb[0:int(trainsize*data.shape[1])]
Hum_amb_test = Hum_amb[int(0.8*data.shape[1]):data.shape[1]]

P_amb = data[ind + 10]
P_amb_train = P_amb[0:int(trainsize*data.shape[1])]
P_amb_test = P_amb[int(0.8*data.shape[1]):data.shape[1]]

x_train.extend((v_windspeed_train, Hum_amb_train, P_amb_train))
x_test.extend((v_windspeed_test, Hum_amb_test, P_amb_test))


"""
-------
x_train
-------
                T_train | SchedVal_train | u_AHU_train  ..... | P_amb_train
sample 1            .   |     .          |     .              |     .
.                   .   |     .          |     .              |     .
sample n            .   |     .          |     .              |     .
"""

x_train = NP.transpose(NP.squeeze(NP.array(x_train)))
x_test = NP.transpose(NP.squeeze(NP.array(x_test)))



"""
Create past values
"""
# x_train
numbers = 2
features = x_train.shape[1]
l = len(T_Coworking_train)
tmp = NP.zeros((l-numbers+1,numbers*features))
for i in range(0,l-numbers+1):
    # tmp[i,:] = x_train[i:i+numbers,0:x_train.shape[1]].transpose().reshape(1,numbers*x_train.shape[1])
    tmp[i,:] = x_train[i:i+numbers,0:features].reshape(1,numbers*features)
x_train = tmp[:]

# x_test
l = len(T_Coworking_test)
tmp = NP.zeros((l-numbers+1,numbers*x_test.shape[1]))
for i in range(0,l-numbers+1):
    # tmp[i,:] = x_test[i:i+numbers,0:x_test.shape[1]].transpose().reshape(1,numbers*x_test.shape[1])
    tmp[i,:] = x_test[i:i+numbers,0:features].reshape(1,numbers*features)
x_test = tmp[:]

# Delete the current T_train(t) for every zone
for i in range(len(zones)):
    x_train = NP.delete(x_train, features*(numbers-1), 1)
    x_test = NP.delete(x_test, features*(numbers-1), 1)



delete_feature = range(len(zones), len(zones) +  len(zones_Heating))
delete_feature.extend(range(len(zones)+len(zones_Heating), len(zones)+len(zones_Heating) + len(zones_ahu)))
delete_feature.extend(range(len(zones)+len(zones_Heating) + len(zones_ahu) + 4, len(zones)+len(zones_Heating) + len(zones_ahu) + 4 + len(zones)))

delete_feature = tuple(delete_feature)
for i in range(features):
    if i in delete_feature:
        x_train = NP.delete(x_train, features, 1)
        x_test = NP.delete(x_test, features, 1)
bp()

"""
Normalize Data
"""
if load == 0:
    x_lb = NP.min(x_train,axis =0)
    x_ub = NP.max(x_train,axis =0)
x_train = NP.divide(x_train - x_lb, x_ub - x_lb)
x_test = NP.divide(x_test - x_lb, x_ub - x_lb)


"""
-------
x_train
-------
                T_train(t-1) |...| SchedVal_train(t-1) | .... | P_amb_train(t-1) | SchedVal_train(t) | u_AHU_train(t)  |...| P_amb_train(t)
sample 1            .        |...|          .          |   .  |      .           |       .           |         .       |...|       .
.                   .        |...|          .          |   .  |      .           |       .           |         .       |...|       .
sample n                     |...|          .          |   .  |      .           |       .           |         .       |...|       .
"""


y_train = NP.transpose(NP.array(y_train))
y_test = NP.transpose(NP.array(y_test))
# throw away the first numbers (no data to train since past)
y_train = y_train[numbers-1:len(T_Coworking_train),:]
y_test = y_test[numbers-1:len(T_Coworking_test),:]


"""
Normalize Data
"""
if load == 0:
    y_lb = NP.min(y_train,axis =0)
    y_ub = NP.max(y_train,axis =0)
y_train = NP.divide(y_train - y_lb, y_ub - y_lb)
y_test = NP.divide(y_test - y_lb, y_ub - y_lb)

# past values are in same row - rows are shuffled here -> keeps current and past values together
NP.random.seed(1)
NP.random.shuffle(x_train)
NP.random.seed(1)
NP.random.shuffle(y_train)

# cast data due to CNTK
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')


if load == 1:
    json_file = open('model_60.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("model_60.h5")
    # model = load_model('NN_model')

else:
    """
    ----------------------------------------
    Definition of the Neural Network
    ----------------------------------------
    """
    model = Sequential()
    model.add(Dense(units = x_train.shape[1], activation='tanh',input_dim = x_train.shape[1]))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh'))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh'))
    #model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh'))
    model.add(Dense(units = y_train.shape[1], activation='linear'))

    Adam = optimizers.Adam(lr=5e-4,decay = 0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(loss='mean_squared_error', optimizer= Adam, metrics = ['mae'])

    """
    ----------------------------------------
    Model Training
    ----------------------------------------
    """
    trained = model.fit(x_train, y_train, validation_split=0, shuffle = False, epochs=1000, batch_size=1024, verbose = 2)
    if save == 1:
        model_json = model.to_json()
        with open("model_60.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_60.h5")
        print("Saved model to disk")
        NP.save('lbub_60.npy', [x_lb, x_ub, y_lb, y_ub])
        # model.save('NN_model')
    # P.subplot(2, 1, 1)
    # P.plot(trained.history['val_loss'] ,'r', label = 'Val loss')
    # P.plot(trained.history['loss'], 'b', label = 'Train loss')
    # P.ylabel('Loss')
    # P.xlabel('epoch')
    #
    # P.subplot(2, 1, 2)
    # P.plot(trained.history['val_mean_absolute_error'] ,'r', label = 'Val MAE')
    # P.plot(trained.history['mean_absolute_error'], 'b', label = 'Train MAE')
    # P.ylabel('MAE')
    # P.xlabel('epoch')
    # P.show()


loss_and_metrics = model.evaluate(x_test, y_test)
print ('loss: ' + str(loss_and_metrics))
classes = NP.squeeze(model.predict(x_test))



y_test = NP.squeeze(y_test)
y_test = NP.multiply(y_test, y_ub-y_lb) + y_lb
classes = NP.multiply(classes, y_ub-y_lb) + y_lb

mse = NP.mean((y_test-classes)**2,axis=0)
rmse = NP.zeros((y_train.shape[1],1))

for i in range(y_train.shape[1]):
    rmse[i] = NP.sqrt(mse[i])
# rmse[2] = NP.sqrt(mse[2])
print("RMSE: " + str(rmse))

# P.subplot(2, 1, 1)
# P.plot(y_test)
# P.plot(classes)
# P.ylabel('SetPoint')
# P.legend(['ZoneTemp','NN'])
#
# P.subplot(2, 1, 2)
# P.plot(u_AHU1_noERC_test)
# P.show()
#
#
# fig2 = P.figure(2)
# P.clf()
# P.plot(y_test- classes)
# P.ylabel('Difference')
# P.show()
