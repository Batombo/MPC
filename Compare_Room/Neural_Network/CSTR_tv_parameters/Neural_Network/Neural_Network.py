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

def relu(x):
    return NP.maximum(0,x)

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

data = NP.load('2_reg.npy')
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
SchedVal = data[0,:]
HeatRate = data[1,:]/100
u_AHU1_noERC = data[2,:]

ZoneTemp = data[3,:]
OutdoorTemp = data[4,:]
v_IG_Offices = data[5,:]
v_solGlobFac_E = data[6,:]
v_solGlobFac_N = data[7,:]
v_solGlobFac_S = data[8,:]
v_solGlobFac_W = data[9,:]
u_blinds_N = data[10,:]
u_blinds_W = data[11,:]
v_windspeed = data[12,:]
Hum_amb = data[13,:]
Hum_zone = data[14,:]

P_amb = data[15,:]




# u_blinds_E[20] = 1e-3
# u_blinds_N[20] = 1e-3
# u_blinds_S[20] = 1e-3
# u_blinds_W[20] = 1e-3



ZoneTempdiff = NP.copy(ZoneTemp)
HeatRatediff = NP.copy(HeatRate)

for i in range(0,len(ZoneTempdiff)-1):
    ZoneTempdiff[i] = ZoneTempdiff[i+1] - ZoneTempdiff[i]
    HeatRatediff[i] = HeatRatediff[i+1] - HeatRatediff[i]

ZoneTempdiff = NP.append(0,ZoneTempdiff[0:-1])
ZoneTempdiff = NP.reshape(ZoneTempdiff, (ZoneTempdiff.shape[0],1))
HeatRatediff = NP.append(0,HeatRatediff[0:-1])
HeatRatediff = NP.reshape(HeatRatediff, (HeatRatediff.shape[0],1))

trainsize = 1

ZoneTempdiff_train = ZoneTempdiff[0:int(trainsize*len(ZoneTempdiff))]
HeatRatediff_train = HeatRatediff[0:int(trainsize*len(HeatRatediff))]

SchedVal_train = SchedVal[0:int(trainsize*len(SchedVal))]
HeatRate_train = HeatRate[0:int(trainsize*len(HeatRate))]
u_AHU1_noERC_train = u_AHU1_noERC[0:int(trainsize*len(u_AHU1_noERC))]
ZoneTemp_train = ZoneTemp[0:int(trainsize*len(ZoneTemp))]
OutdoorTemp_train = OutdoorTemp[0:int(trainsize*len(OutdoorTemp))]
v_IG_Offices_train = v_IG_Offices[0:int(trainsize*len(v_IG_Offices))]
v_solGlobFac_E_train = v_solGlobFac_E[0:int(trainsize*len(v_solGlobFac_E))]
v_solGlobFac_N_train = v_solGlobFac_N[0:int(trainsize*len(v_solGlobFac_N))]
v_solGlobFac_S_train = v_solGlobFac_S[0:int(trainsize*len(v_solGlobFac_S))]
v_solGlobFac_W_train = v_solGlobFac_W[0:int(trainsize*len(v_solGlobFac_W))]
u_blinds_N_train = u_blinds_N[0:int(trainsize*len(u_blinds_N))]
u_blinds_W_train = u_blinds_W[0:int(trainsize*len(u_blinds_W))]
v_windspeed_train = v_windspeed[0:int(trainsize*len(v_windspeed))]
Hum_amb_train = Hum_amb[0:int(trainsize*len(Hum_amb))]
Hum_zone_train = Hum_zone[0:int(trainsize*len(Hum_zone))]

P_amb_train = P_amb[0:int(trainsize*len(Hum_zone))]

# Normalize Data
"""
-------
x_train
-------
                HeatRate | ZoneTemp | OutdoorTemp
sample 1            .    |     .    |     .
.                   .    |     .    |     .
sample n            .    |     .    |     .
"""
x_train = NP.transpose(NP.squeeze(NP.array([ZoneTemp_train,# HeatRate_train,
u_blinds_N_train, u_blinds_W_train,
u_AHU1_noERC_train, SchedVal_train,
v_IG_Offices_train, OutdoorTemp_train,
v_solGlobFac_E_train, v_solGlobFac_N_train, v_solGlobFac_S_train, v_solGlobFac_W_train,
v_windspeed_train, Hum_amb_train, Hum_zone_train, P_amb_train])))


numbers = 2
features = x_train.shape[1]
l = len(ZoneTemp_train)
tmp = NP.zeros((l-numbers+1,numbers*features))
for i in range(0,l-numbers+1):
    # tmp[i,:] = x_train[i:i+numbers,0:x_train.shape[1]].transpose().reshape(1,numbers*x_train.shape[1])
    tmp[i,:] = x_train[i:i+numbers,0:features].reshape(1,numbers*features)
x_train = tmp[:]
# Delete the current ZoneTemp(t)
x_train = NP.delete(x_train, features*(numbers-1), 1)
# x_train = NP.delete(x_train, features*(numbers-1)+1-1, 1) # -1 since already one element missing
# x_train = NP.delete(x_train, features*(numbers-1)+1-1, 1) # -1 since already one element missing
if load == 0:
    x_lb = NP.min(x_train,axis =0)
    x_ub = NP.max(x_train,axis =0)
x_train = NP.divide(x_train - x_lb, x_ub - x_lb)

"""
-------
x_train
-------
                ZoneTemp(t-1) |...| HeatRate(t) | ZoneTemp(t-n+1) |...| ZoneTemp(t) | OutdoorTemp(t-n+1) |...| OutdoorTemp(t)
sample 1            .          |...|     .       |       .        |...|     .       |         .         |...|       .
.                   .          |...|     .       |       .        |...|     .       |         .         |...|       .
sample n                       |...|     .       |       .        |...|     .       |         .         |...|       .
"""


y_train = NP.transpose(NP.array([ZoneTemp_train[numbers-1:len(ZoneTemp_train)], HeatRate_train[numbers-1:len(HeatRate_train)]]))
# y_train = NP.reshape(y_train, (y_train.shape[1],y_train.shape[2]))
if load == 0:
    y_lb = NP.min(y_train,axis =0)
    y_ub = NP.max(y_train,axis =0)
y_train = NP.divide(y_train - y_lb, y_ub - y_lb)
NP.random.seed(1)
NP.random.shuffle(x_train)
NP.random.seed(1)
NP.random.shuffle(y_train)



if load == 1:
    #
    json_file = open('model_3.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("model_3.h5")
    # Theta = {}
    # for i in range(len(model.layers)):
    #     weights = model.layers[i].get_weights()
    #     Theta['Theta'+str(i+1)] =  NP.insert(weights[0].transpose(),0,weights[1].transpose(),axis=1)
    model = load_model('NN_model')

else:
    """
    ----------------------------------------
    Definition of the Neural Network
    ----------------------------------------
    """
    drate = 0.0
    model = Sequential()
    model.add(Dense(units = x_train.shape[1], activation='tanh',input_dim = x_train.shape[1]))
    model.add(Dropout(drate))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh'))
    model.add(Dropout(drate))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh'))
    model.add(Dropout(drate))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh'))
    model.add(Dropout(drate))
    # model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='relu'))

    model.add(Dense(units = 2, activation='linear'))

    Adam = optimizers.Adam(lr=5e-4,decay = 0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(loss='mean_squared_error', optimizer= Adam, metrics = ['mae'])

    """
    ----------------------------------------
    Model Training
    ----------------------------------------
    """
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    trained = model.fit(x_train, y_train, validation_split=0.1, shuffle = False, epochs=1000, batch_size=1024, verbose = 2)
    if save == 1:
        model_json = model.to_json()
        with open("model_3.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_3.h5")
        print("Saved model to disk")
        NP.save('lbub.npy', [x_lb, x_ub, y_lb, y_ub])
        model.save('NN_model')
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
"""
----------------------------------------
Test Data Prep
----------------------------------------
"""

ZoneTempdiff_test = ZoneTempdiff[int(0.8*len(ZoneTempdiff))+1:len(ZoneTempdiff)]
HeatRatediff_test = HeatRatediff[int(0.8*len(HeatRatediff))+1:len(HeatRatediff)]

SchedVal_test = SchedVal[int(0.8*len(SchedVal))+1:len(SchedVal)]
HeatRate_test = HeatRate[int(0.8*len(HeatRate))+1:len(HeatRate)]
u_AHU1_noERC_test = u_AHU1_noERC[int(0.8*len(u_AHU1_noERC))+1:len(u_AHU1_noERC)]
ZoneTemp_test = ZoneTemp[int(0.8*len(ZoneTemp))+1:len(ZoneTemp)]
OutdoorTemp_test = OutdoorTemp[int(0.8*len(OutdoorTemp))+1:len(OutdoorTemp)]


v_IG_Offices_test = v_IG_Offices[int(0.8*len(v_IG_Offices))+1:len(v_IG_Offices)]

v_solGlobFac_E_test = v_solGlobFac_E[int(0.8*len(v_solGlobFac_E))+1:len(v_solGlobFac_E)]
v_solGlobFac_N_test = v_solGlobFac_N[int(0.8*len(v_solGlobFac_N))+1:len(v_solGlobFac_N)]
v_solGlobFac_S_test = v_solGlobFac_S[int(0.8*len(v_solGlobFac_S))+1:len(v_solGlobFac_S)]
v_solGlobFac_W_test = v_solGlobFac_W[int(0.8*len(v_solGlobFac_W))+1:len(v_solGlobFac_W)]

u_blinds_N_test = u_blinds_N[int(0.8*len(u_blinds_N))+1:len(u_blinds_N)]
u_blinds_W_test = u_blinds_W[int(0.8*len(u_blinds_W))+1:len(u_blinds_W)]

v_windspeed_test = v_windspeed[int(0.8*len(v_windspeed))+1:len(v_windspeed)]
Hum_amb_test = Hum_amb[int(0.8*len(Hum_amb))+1:len(Hum_amb)]
Hum_zone_test = Hum_zone[int(0.8*len(Hum_zone))+1:len(Hum_zone)]

P_amb_test = P_amb[int(0.8*len(Hum_zone))+1:len(Hum_zone)]

# do normalization with same mu and var as with training set

x_test = NP.transpose(NP.squeeze(NP.array([ZoneTemp_test,# HeatRate_test,
u_blinds_N_test, u_blinds_W_test,
u_AHU1_noERC_test, SchedVal_test,
v_IG_Offices_test, OutdoorTemp_test,
v_solGlobFac_E_test, v_solGlobFac_N_test, v_solGlobFac_S_test, v_solGlobFac_W_test,
v_windspeed_test, Hum_amb_test, Hum_zone_test, P_amb_test])))


features = x_test.shape[1]
l = len(SchedVal_test)
tmp = NP.zeros((l-numbers+1,numbers*x_test.shape[1]))
for i in range(0,l-numbers+1):
    # tmp[i,:] = x_test[i:i+numbers,0:x_test.shape[1]].transpose().reshape(1,numbers*x_test.shape[1])
    tmp[i,:] = x_test[i:i+numbers,0:features].reshape(1,numbers*features)
x_test = tmp[:]

# Delete the current ZoneTemp(t)
x_test = NP.delete(x_test, features*(numbers-1), 1)
# x_test = NP.delete(x_test, features*(numbers-1)+1-1, 1) # -1 since already one element missing
# x_test = NP.delete(x_test, features*(numbers-1)+1-1, 1) # -1 since already one element missing
x_test = NP.divide(x_test - x_lb, x_ub - x_lb)


y_test = NP.transpose(NP.array([ZoneTemp_test[numbers-1:len(ZoneTemp_test)], HeatRate_test[numbers-1:len(HeatRate_test)]]))
# y_test = NP.reshape(y_test, (y_test.shape[1],y_test.shape[2]))
y_test = NP.divide(y_test - y_lb, y_ub - y_lb)

x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=512)
print ('loss: ' + str(loss_and_metrics))
classes = NP.squeeze(model.predict(x_test))#, batch_size=512)



y_test = NP.squeeze(y_test)
y_test = NP.multiply(y_test, y_ub-y_lb) + y_lb
classes = NP.multiply(classes, y_ub-y_lb) + y_lb

mse = NP.mean((y_test-classes)**2,axis=0)
rmse = NP.zeros((3,1))

rmse[0] = NP.sqrt(mse[0])
rmse[1] = NP.sqrt(mse[1])
# rmse[2] = NP.sqrt(mse[2])
# print("RMSE: " + str(rmse))
#
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
