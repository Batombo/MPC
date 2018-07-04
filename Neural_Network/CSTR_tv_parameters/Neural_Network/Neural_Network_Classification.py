
"""
------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------
"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import model_from_json
from keras import optimizers, regularizers

import scipy.io as sio
import numpy as NP
from pdb import set_trace as bp
import random
import pylab as P
import time



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

data = NP.load('full.npy')
SchedVal = shuffleMonths(data[0,:])
HeatRate = shuffleMonths(data[1,:])
ZoneTemp = shuffleMonths(data[2,:])
OutdoorTemp = shuffleMonths(data[3,:])


# Create Classification Problem
minval = int(NP.min(SchedVal))
r = int(NP.max(SchedVal) - minval) + 1
tmp = NP.zeros((len(SchedVal), r))
for i in range(0,r):
    val =  i + NP.min(SchedVal)
    tmp[NP.squeeze(SchedVal == val), i] = 1
SchedVal = tmp[:]

HeatRate_train = HeatRate[0:int(0.8*len(HeatRate))]
ZoneTemp_train = ZoneTemp[0:int(0.8*len(ZoneTemp))]
OutdoorTemp_train = OutdoorTemp[0:int(0.8*len(OutdoorTemp))]
SchedVal_train = SchedVal[0:int(0.8*len(SchedVal))]


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
x_train = NP.transpose(NP.squeeze([HeatRate_train, ZoneTemp_train, OutdoorTemp_train]))
# mu = NP.mean(x_train, axis = 0)
# x_train -= mu
# var = NP.var(x_train, axis = 0)
# x_train = NP.divide(x_train, var**0.5)

numbers = 4*6
l = len(SchedVal_train)
bp()
tmp = NP.zeros((l-numbers+1,numbers*x_train.shape[1]))
for i in range(0,l-numbers+1):
    tmp[i,:] = x_train[i:i+numbers,0:x_train.shape[1]].transpose().reshape(1,numbers*x_train.shape[1])
x_train = tmp[:]
"""
-------
x_train
-------
                HeatRate(t-23) |...| HeatRate(t) | ZoneTemp(t-23) |...| ZoneTemp(t) | OutdoorTemp(t-23) |...| OutdoorTemp(t)
sample 1            .          |...|     .       |       .        |...|     .       |         .         |...|       .
.                   .          |...|     .       |       .        |...|     .       |         .         |...|       .
sample n                       |...|     .       |       .        |...|     .       |         .         |...|       .
"""

bp()
y_train = SchedVal_train[numbers-1:len(SchedVal_train)]
NP.random.seed(1)
NP.random.shuffle(x_train)
NP.random.seed(1)
NP.random.shuffle(y_train)
#
# fig = P.figure(1)
# P.clf()
# P.plot(OutdoorTemp_train)
# P.ylabel('SetPoint')
# P.show()



load = 1
save = 1
if load == 1:
    model = load_model('NN_model')


    model_json = model.to_json()
    with open("model_3.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_3.h5")
    print("Saved model to disk")


    json_file = open('model_3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_3.h5")
    print("Loaded model from disk")
    model = loaded_model
else:
    """
    ----------------------------------------
    Definition of the Neural Network
    ----------------------------------------
    """

    drate = 0
    lrate = 0#2e-5
    lrate2 = 0#2e-5
    model = Sequential()
    model.add(Dense(units = x_train.shape[1], activation='tanh',input_dim = x_train.shape[1], kernel_regularizer=regularizers.l2(lrate2)))
    # model.add(Dropout(drate))

    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh', kernel_regularizer=regularizers.l2(lrate)))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh', kernel_regularizer=regularizers.l2(lrate)))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh', kernel_regularizer=regularizers.l2(lrate)))
    ######
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh', kernel_regularizer=regularizers.l2(lrate)))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh', kernel_regularizer=regularizers.l2(lrate)))
    ######
    model.add(Dense(units = SchedVal_train.shape[1], activation='softmax'))
    # model.add(Dropout(drate))
    #
    # model.add(Dense(units = 1, activation='linear'))
#8e-4
    Adam = optimizers.Adam(lr=1e-3,decay = 0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    SGD = optimizers.SGD(lr=0.05, momentum=0.9, decay=0.1, nesterov=True)
    RMSprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer= Adam, metrics = ['accuracy'])

    """
    ----------------------------------------
    Model Training
    ----------------------------------------
    """
    trained = model.fit(x_train, y_train, validation_split=0.1, shuffle = False, epochs=500, batch_size=128, verbose = 2)
    if save == 1:
        model.save('NN_model')
    P.subplot(2, 1, 1)
    P.plot(trained.history['val_loss'] ,'r', label = 'Val loss')
    P.plot(trained.history['loss'], 'b', label = 'Train loss')
    P.ylabel('Loss')
    P.xlabel('epoch')


    P.subplot(2, 1, 2)
    P.plot(trained.history['val_acc'] ,'r', label = 'Val acc')
    P.plot(trained.history['acc'], 'b', label = 'Train acc')
    P.ylabel('Acc')
    P.xlabel('epoch')
    P.show()


"""
----------------------------------------
Test Data Prep
----------------------------------------
"""

HeatRate_test = HeatRate[int(0.8*len(HeatRate))+1:len(HeatRate)]
ZoneTemp_test = ZoneTemp[int(0.8*len(ZoneTemp))+1:len(ZoneTemp)]
OutdoorTemp_test = OutdoorTemp[int(0.8*len(OutdoorTemp))+1:len(OutdoorTemp)]
SchedVal_test = SchedVal[int(0.8*len(SchedVal))+1:len(SchedVal)]


# do normalization with same mu and var as with training set
x_test = NP.transpose(NP.squeeze([HeatRate_test, ZoneTemp_test, OutdoorTemp_test]))
x_test -= mu
x_test = NP.divide(x_test, var**0.5)

NP.save('muvar.npy', [mu, var])

l = len(SchedVal_test)
tmp = NP.zeros((l-numbers+1,numbers*x_test.shape[1]))
for i in range(0,l-numbers+1):
    tmp[i,:] = x_test[i:i+numbers,0:x_test.shape[1]].transpose().reshape(1,numbers*x_test.shape[1])
x_test = tmp[:]
y_test = SchedVal_test[numbers-1:len(SchedVal_train)]


# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=512)
# print ('loss: ' + str(loss_and_metrics))

start_time = time.time()
classes = model.predict(x_test, batch_size=512)
elapsed_time = time.time() - start_time
print(elapsed_time)
bp()
start_time = time.time()
classes = model.predict(x_test, batch_size=512)
elapsed_time = time.time() - start_time
print(elapsed_time)
bp()

tmp_class = NP.argmax(classes,axis=1)
tmp_sched = NP.argmax(y_test,axis=1)
class_res = NP.zeros(len(y_test))
sched_res = NP.zeros(len(y_test))
for i in range(0,len(y_test)):
    class_res[i] = tmp_class[i] + minval
    sched_res[i] = tmp_sched[i] + minval


rmse = NP.sqrt(NP.mean((sched_res-class_res)**2))
print("RMSE: " + str(rmse))

fig = P.figure(1)
P.clf()
P.plot(sched_res)
P.plot(class_res)
P.ylabel('SetPoint')
P.show()


fig2 = P.figure(2)
P.clf()
P.plot(sched_res- class_res)
P.ylabel('Difference')
P.show()
