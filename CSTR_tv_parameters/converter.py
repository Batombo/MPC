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

json_file = open('Neural_Network\model_3.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("Neural_Network\model_3.h5")
print("Loaded model from disk")

lbub = NP.load('Neural_Network\lbub.npy')
x_lb = lbub[0]
x_ub = lbub[1]
y_lb = lbub[2]
y_ub = lbub[3]

def HeatingRate2Schedule(x_test):
    x_test = x_test.reshape(1,len(x_test))
    x_test = NP.divide(x_test - x_lb, x_ub - x_lb)
    classes = model.predict(x_test)#, batch_size = 512)
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
