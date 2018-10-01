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



"""
----------------------------------------
Training Data Prep
----------------------------------------
"""

data = NP.load('full.npy')

load = 1
save = 0

if load == 1:
    lbub = NP.load('lbub.npy')
    x_lb = lbub[0]
    x_ub = lbub[1]
    y_lb = lbub[2]
    y_ub = lbub[3]

data = NP.load('full.npy')
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
P_amb = data[14,:]


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
P_amb_train = P_amb[0:int(trainsize*len(P_amb))]

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
v_windspeed_train, Hum_amb_train, P_amb_train])))


numbers = 2
features = x_train.shape[1]
l = len(ZoneTemp_train)
tmp = NP.zeros((l-numbers+1,numbers*features))
for i in range(0,l-numbers+1):
    tmp[i,:] = x_train[i:i+numbers,0:features].reshape(1,numbers*features)
x_train = tmp[:]
# Delete the current ZoneTemp(t)
x_train = NP.delete(x_train, features*(numbers-1), 1)

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
    json_file = open('model_full.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("model_full.h5")


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
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh'))

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
    trained = model.fit(x_train, y_train, validation_split=0, shuffle = False, epochs=1500, batch_size=1024, verbose = 2)
    if save == 1:
        model_json = model.to_json()
        with open("model_full.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_full.h5")
        print("Saved model to disk")
        NP.save('lbub.npy', [x_lb, x_ub, y_lb, y_ub])
"""
----------------------------------------
Test Data Prep
----------------------------------------
"""

ZoneTempdiff_test = ZoneTempdiff[int(0.8*len(ZoneTempdiff))+1:len(ZoneTempdiff)]
HeatRatediff_test = HeatRatediff[int(0.8*len(HeatRatediff))+1:len(HeatRatediff)]


data = NP.load('test_data.npy')
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
P_amb = data[14,:]




st = 160
days = 30
daypoints = 144


SchedVal_test = SchedVal[daypoints*st:daypoints*(st+days)]
HeatRate_test = HeatRate[daypoints*st:daypoints*(st+days)]
u_AHU1_noERC_test = u_AHU1_noERC[daypoints*st:daypoints*(st+days)]
ZoneTemp_test = ZoneTemp[daypoints*st:daypoints*(st+days)]
OutdoorTemp_test = OutdoorTemp[daypoints*st:daypoints*(st+days)]


v_IG_Offices_test = v_IG_Offices[daypoints*st:daypoints*(st+days)]

v_solGlobFac_E_test = v_solGlobFac_E[daypoints*st:daypoints*(st+days)]
v_solGlobFac_N_test = v_solGlobFac_N[daypoints*st:daypoints*(st+days)]
v_solGlobFac_S_test = v_solGlobFac_S[daypoints*st:daypoints*(st+days)]
v_solGlobFac_W_test = v_solGlobFac_W[daypoints*st:daypoints*(st+days)]

u_blinds_N_test = u_blinds_N[daypoints*st:daypoints*(st+days)]
u_blinds_W_test = u_blinds_W[daypoints*st:daypoints*(st+days)]

v_windspeed_test = v_windspeed[daypoints*st:daypoints*(st+days)]
Hum_amb_test = Hum_amb[daypoints*st:daypoints*(st+days)]

P_amb_test = P_amb[daypoints*st:daypoints*(st+days)]

# do normalization with same mu and var as with training set

x_test = NP.transpose(NP.squeeze(NP.array([ZoneTemp_test,# HeatRate_test,
u_blinds_N_test, u_blinds_W_test,
u_AHU1_noERC_test, SchedVal_test,
v_IG_Offices_test, OutdoorTemp_test,
v_solGlobFac_E_test, v_solGlobFac_N_test, v_solGlobFac_S_test, v_solGlobFac_W_test,
v_windspeed_test, Hum_amb_test, P_amb_test])))


features = x_test.shape[1]
l = len(SchedVal_test)
tmp = NP.zeros((l-numbers+1,numbers*x_test.shape[1]))
for i in range(0,l-numbers+1):
    # tmp[i,:] = x_test[i:i+numbers,0:x_test.shape[1]].transpose().reshape(1,numbers*x_test.shape[1])
    tmp[i,:] = x_test[i:i+numbers,0:features].reshape(1,numbers*features)
x_test = tmp[:]

# Delete the current ZoneTemp(t)
x_test = NP.delete(x_test, features*(numbers-1), 1)

x_test = NP.divide(x_test - x_lb, x_ub - x_lb)


y_test = NP.transpose(NP.array([ZoneTemp_test[numbers-1:len(ZoneTemp_test)], HeatRate_test[numbers-1:len(HeatRate_test)]]))

y_test = NP.divide(y_test - y_lb, y_ub - y_lb)

x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

classes = NP.squeeze(model.predict(x_test))



y_test = NP.squeeze(y_test)
y_test = NP.multiply(y_test, y_ub-y_lb) + y_lb
classes = NP.multiply(classes, y_ub-y_lb) + y_lb

mse = NP.mean((y_test-classes)**2,axis=0)
rmse = NP.zeros((2,1))

rmse[0] = NP.sqrt(mse[0])
rmse[1] = NP.sqrt(mse[1])

print("RMSE: " + str(rmse))

np.save('results_ANN.npy', [y_test[:,0], classes[:,0], rmse[0]])


P.subplot(2, 1, 1)
P.plot(y_test[:,0])
P.plot(classes[:,0])
P.ylabel('SetPoint')
P.legend(['ZoneTemp','NN'])

P.subplot(2, 1, 2)
P.plot(OutdoorTemp_test)
P.show()


fig2 = P.figure(2)
P.clf()
P.plot(y_test- classes)
P.ylabel('Difference')
P.show()
