from pdb import set_trace as bp
import scipy.io as sio
import numpy as NP
import pylab as P
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import model_from_json
from keras import optimizers, regularizers

# data = NP.load('full10.npy')
#
# HeatRate = data[1,:]
# ZoneTemp = data[2,:]
# OutdoorTemp = data[3,:]
# SchedVal = data[0,:]
# HeatRate_test = HeatRate[int(round(0.8*len(HeatRate))+1):len(HeatRate)]
# ZoneTemp_test = ZoneTemp[int(round(0.8*len(ZoneTemp))+1):len(ZoneTemp)]
# OutdoorTemp_test = OutdoorTemp[int(round(0.8*len(OutdoorTemp))+1):len(OutdoorTemp)]

# def HeatingRate2Schedule(HeatRate, ZoneTemp, OutdoorTemp):
#     x_test = NP.transpose(NP.squeeze([HeatRate, ZoneTemp, OutdoorTemp]))
#     NP.save('x_test.npy', [x_test])
#     child = subprocess.call('python3 Neural_Network.py' ,shell=True)
#     class_res = NP.load('class_res.npy')
#     class_res =  NP.transpose(class_res)
#
#     return class_res

#
# muvar = NP.load('muvar.npy')
# mu = muvar[0,:]
# var = muvar[1,:]
#
#
# json_file = open('model_3.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model_3.h5")
# print("Loaded model from disk")
# model = loaded_model

lbub = NP.load('lbub.npy')
x_lb = lbub[0]
x_ub = lbub[1]
y_lb = lbub[2]
y_ub = lbub[3]

model = load_model('NN_model')


def HeatingRate2Schedule(x_test):

    # x_test -= mu
    # x_test = NP.divide(x_test, var**0.5)
    x_test = x_test.reshape(1,len(x_test))
    x_test = NP.divide(x_test - x_lb, x_ub - x_lb)
    # x_test = x_test.transpose().reshape(1,71)
    classes = model.predict(x_test, batch_size = 512)
    classes = NP.multiply(classes, y_ub-y_lb) + y_lb

    return NP.round(classes, 2)

def massflow2volumeflow(temperature, massflow):
    # returns the Density of air
    # coefficients from regression
    p = NP.array([1.435714285714305e-05, -0.004690357142857, 1.292137500000000])
    temp = NP.array([temperature**2, temperature, 1])
    rho = NP.matmul(p, temp)
    return NP.divide(massflow, rho)


def volumeflow2sched(volumeflow):
    return NP.divide(volumeflow, 0.16)
