from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras import optimizers, regularizers

import scipy.io
import numpy as NP
from pdb import set_trace as bp
import random
import pylab as P


numbers = 3*6

"""
----------------------------------------
Training Data Prep
----------------------------------------
"""

# data = scipy.io.loadmat('Data.mat')
data = NP.load('full.npy')
SchedVal = data[0,:]
HeatRate = data[1,:]
ZoneTemp = data[2,:]
OutdoorTemp = data[3,:]
bp()
# Shuffle the data
n = numbers
HeatRate = [HeatRate[i:i + n] for i in range(0, len(HeatRate), n)]
HeatRate = NP.squeeze(NP.asarray(HeatRate))
random.seed(0)
random.shuffle(HeatRate)
HeatRate = NP.reshape(HeatRate,(HeatRate.shape[0]*HeatRate.shape[1],1))

ZoneTemp = [ZoneTemp[i:i + n] for i in range(0, len(ZoneTemp), n)]
ZoneTemp = NP.squeeze(NP.asarray(ZoneTemp))
random.seed(0)
random.shuffle(ZoneTemp)
ZoneTemp = NP.reshape(ZoneTemp,(ZoneTemp.shape[0]*ZoneTemp.shape[1],1))

OutdoorTemp = [OutdoorTemp[i:i + n] for i in range(0, len(OutdoorTemp), n)]
OutdoorTemp = NP.squeeze(NP.asarray(OutdoorTemp))
random.seed(0)
random.shuffle(OutdoorTemp)
OutdoorTemp = NP.reshape(OutdoorTemp,(OutdoorTemp.shape[0]*OutdoorTemp.shape[1],1))


SchedVal = [SchedVal[i:i + n] for i in range(0, len(SchedVal), n)]
SchedVal = NP.squeeze(NP.asarray(SchedVal))
random.seed(0)
random.shuffle(SchedVal)
SchedVal = NP.reshape(SchedVal,(SchedVal.shape[0]*SchedVal.shape[1],1))


# create a classification problem
minval = int(NP.min(SchedVal))
r = int(NP.max(SchedVal) - minval) + 1
tmp = NP.zeros((len(SchedVal), r))
for i in range(0,r):
    val =  i + NP.min(SchedVal)
    tmp[NP.squeeze(SchedVal == val), i] = 1
SchedVal = tmp[:]





#NOTE can do Batch Norm?

HeatRate_train = HeatRate[0:round(0.8*len(HeatRate))]
ZoneTemp_train = ZoneTemp[0:round(0.8*len(ZoneTemp))]
OutdoorTemp_train = OutdoorTemp[0:round(0.8*len(OutdoorTemp))]
SchedVal_train = SchedVal[0:round(0.8*len(SchedVal))]


# Normalize Data
x_train = NP.transpose(NP.squeeze([HeatRate_train, ZoneTemp_train, OutdoorTemp_train]))
mu = NP.mean(x_train, axis = 0)
x_train -= mu
var = NP.var(x_train, axis = 0)
x_train = NP.divide(x_train, var**0.5)

past = {}
past_y = {}

#FIXME Merge the two for loops together
for i in range(1,numbers):
    # Take the 'orignal' three variables
    past['x_train' + str(i)] = x_train[i:len(x_train)-numbers+i, 0:3]
    past_y['y_train' + str(i)] = SchedVal_train[i:len(SchedVal_train)-numbers+i, :]

# cut out variables
x_train = x_train[numbers:len(x_train)]
SchedVal_train = SchedVal_train[numbers:len(SchedVal_train)]

# # merge together 'past and actual' variables
for i in range(1,numbers):
    x_train = NP.append(x_train, past['x_train' + str(i)], axis = 1)
    SchedVal_train = NP.append(SchedVal_train, past_y['y_train' + str(i)], axis = 1)

# reshape input to be 3D [samples, timesteps, features]
var = NP.zeros((x_train.shape[0], numbers + 1, 3))
var_y =  NP.zeros((SchedVal_train.shape[0], numbers + 1, 14))
for i in range(0,numbers):
    var[:,i,:] = x_train[:,3*i:3*i+3]
    var_y[:,i,:] = SchedVal_train[:,14*i:14*i+14]

x_train = var
SchedVal_train = var_y


load = 0
save = 0
if load == 1:
    model = load_model('NN_model_Classification')
else:
    lrate = NP.zeros(6)
    drate = NP.zeros(6)
    """
    ----------------------------------------
    Definition of the Neural Network
    ----------------------------------------
    """
    model = Sequential()
    model.add(LSTM(activation = 'tanh',units = x_train.shape[1],unit_forget_bias = True, return_sequences = True,input_shape = (x_train.shape[1], 3), kernel_regularizer=regularizers.l2(0.0000)))
    # model.add(LSTM(activation = 'tanh',units = x_train.shape[1],unit_forget_bias = True, return_sequences = True, kernel_regularizer=regularizers.l2(0.0002)))
    # model.add(LSTM(activation = 'tanh',units = x_train.shape[1],unit_forget_bias = True, return_sequences = True))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh', kernel_regularizer=regularizers.l2(0)))
    model.add(Dense(units = int(NP.round(x_train.shape[1]/1)), activation='tanh', kernel_regularizer=regularizers.l2(0)))
    model.add(Dense(activation='softmax', units = 14))

#8e-4
    Adam = optimizers.Adam(lr=1e-2,decay = 0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='mean_squared_logarithmic_error', optimizer= Adam, metrics = ['accuracy'])

    """
    ----------------------------------------
    Model Training
    ----------------------------------------
    """

    # Reshape Arrays for LSTM
    # x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1] ))
    # SchedVal_train = SchedVal_train[numbers:SchedVal_train.shape[0]]
    # reshape input to be 3D [samples, timesteps, features]
    # SchedVal_train = SchedVal_train.reshape((SchedVal_train.shape[0],1,SchedVal_train.shape[1]))
    trained = model.fit(x_train,SchedVal_train , validation_split=0.2, shuffle = False, epochs=1000, batch_size= 2048, verbose = 2)

    if save == 1:
        model.save('NN_model_Classification')

    P.subplot(2, 1, 1)
    P.plot(trained.history['val_loss'] ,'r', label = 'Val loss')
    P.plot(trained.history['loss'], 'b', label = 'Train loss')
    P.ylabel('Loss')
    P.xlabel('epoch')


    P.subplot(2, 1, 2)
    P.plot(trained.history['val_acc'] ,'r', label = 'Val acc')
    P.plot(trained.history['acc'], 'b', label = 'Train acc')
    P.ylabel('MAE')
    P.xlabel('epoch')
    P.show()


"""
----------------------------------------
Test Data Prep
----------------------------------------
"""

HeatRate_test = HeatRate[round(0.8*len(HeatRate))+1:len(HeatRate)]
ZoneTemp_test = ZoneTemp[round(0.8*len(ZoneTemp))+1:len(ZoneTemp)]
OutdoorTemp_test = OutdoorTemp[round(0.8*len(OutdoorTemp))+1:len(OutdoorTemp)]
SchedVal_test = SchedVal[round(0.8*len(SchedVal))+1:len(SchedVal)]


# do normalization with same mu and var as with training set
x_test = NP.transpose(NP.squeeze([HeatRate_test, ZoneTemp_test, OutdoorTemp_test]))
x_test -= mu
x_test = NP.divide(x_test, var**0.5)


past_test = {}
#FIXME Merge the two for loops together
for i in range(0,numbers):
    # Take the 'orignal' three variables
    past_test['x_test' + str(i)] = x_test[i:len(x_test)-numbers+i, 1:3]

# cut out variables
x_test = x_test[numbers:len(x_test)]

# merge together 'past and actual' variables
for i in range(0,numbers):
    x_test = NP.append(x_test, past_test['x_test' + str(i)], axis = 1)


x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1] ))
SchedVal_test = SchedVal_test[numbers:SchedVal_test.shape[0]]
SchedVal_test = SchedVal_test.reshape((SchedVal_test.shape[0],1,SchedVal_test.shape[1]))

loss_and_metrics = model.evaluate(x_test, SchedVal_test, batch_size=2048)
print ('loss: ' + str(loss_and_metrics))

# classes = model.predict(x_test, batch_size=2048)

# SchedVal_test = SchedVal_test[numbers:len(SchedVal_test)]
classes = classes.reshape(classes.shape[0], classes.shape[2])
SchedVal_test = SchedVal_test.reshape(SchedVal_test.shape[0], SchedVal_test.shape[2])
tmp_class = NP.argmax(classes,axis=1)
tmp_sched = NP.argmax(SchedVal_test,axis=1)
class_res = NP.zeros(len(SchedVal_test))
sched_res = NP.zeros(len(SchedVal_test))
for i in range(0,len(SchedVal_test)):
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
