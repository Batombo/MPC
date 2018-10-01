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

"""
----------------------------------------
Training Data Prep
----------------------------------------
"""

load = 1
save = 0
trainsize = 1
if load == 0:
    data = NP.load('full.npy')
else:
    data = NP.load('Berlin.npy')


# retrieve simulated variables from FMU.py
# create training & test data
x_test = []
y_test = []
x_train = []
y_train = []

# not all zones have windows or radiators - # NOTE: Elevator is not controlled therefore not in zones list
# NOTE: this list needs to be in the same order as specified in FMU.py
zones = ['Coworking', 'Corridor', 'Entrance', 'LabNorth', 'LabSouth', 'MeetingSouth',
'MeetingNorth', 'Nerdroom1', 'Nerdroom2', 'RestroomM', 'RestroomW',
'Space01', 'Stairway']

eastSide = ['Coworking', 'Space01', 'Stairway', 'RestroomW', 'Corridor']
northSide = ['Coworking', 'MeetingNorth', 'LabNorth']
southSide = ['Nerdroom2', 'Nerdroom1', 'MeetingSouth', 'LabSouth']
westSide = ['LabSouth', 'Entrance', 'LabNorth']

# you may want to change the ind depending on which zone you want to train
# e.g., for training zone 'Entrance' change to ind = 2*5
ind = 0*5
for zone in zones:
    vars()['Heatrate_' + zone] = data[0 + ind,:]/100
    vars()['T_' + zone] = data[1 + ind,:]
    vars()['SchedVal_' + zone] = data[2 + ind,:]
    if zone == 'RestroomM':
        vars()['u_AHU_' + zone] = 1*data[3 + 10*5,:]
    else:
        vars()['u_AHU_' + zone] = data[3 + ind,:]
    vars()['u_AHU_' + zone][23] = 1e-3
    vars()['v_IG_' + zone] = data[4 + ind,:]
    ind += 5
# make sure the ind is correct for zone independent data
ind = 65
v_Tamb = data[ind,:]

v_solGlobFac_E = data[ind+1,:]
v_solGlobFac_N = data[ind+2,:]
v_solGlobFac_S = data[ind+3,:]
v_solGlobFac_W = data[ind+4,:]

v_windspeed = data[ind+9,:]
Hum_amb = data[ind+10,:]
P_amb = data[ind+11,:]

# one past value one current value
numbers = 2
# train every Network for every zone
for zone in zones:
    u_blinds_E = data[ind+5,:]
    u_blinds_N = data[ind+6,:]
    u_blinds_S = data[ind+7,:]
    u_blinds_W = data[ind+8,:]

    # train networks with dummy features (this is a dirty workaround) so you
    # dont have to care about the ANN input size in template_model - they all
    # the same
    if zone not in eastSide:
        u_blinds_E = 0*data[ind+5,:]
        u_blinds_E[23] = 1e-3
    if zone not in northSide:
        u_blinds_N = 0*data[ind+6,:]
        u_blinds_N[23] = 1e-3
    if zone not in southSide:
        u_blinds_S = 0*data[ind+7,:]
        u_blinds_S[23] = 1e-3
    if zone not in westSide:
        u_blinds_W = 0*data[ind+8,:]
        u_blinds_W[23] = 1e-3

    x = NP.transpose(NP.squeeze(NP.array([
    vars()['T_'+zone],
    u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W,
    vars()['u_AHU_'+zone], vars()['SchedVal_'+zone],
    vars()['v_IG_'+zone], v_Tamb,
    v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W,
    v_windspeed, Hum_amb, P_amb])))
    # define training and test data - these do not incorporate past data yet
    x_train = x[0:int(trainsize*len(x))]
    x_test = x[int(0.0*len(x)):]
    """
    -------
    x_train
    -------
                    T_train | SchedVal_train | u_AHU_train  ..... | P_amb_train
    sample 1            .   |     .          |     .              |     .
    .                   .   |     .          |     .              |     .
    sample n            .   |     .          |     .              |     .
    """
    y_train = NP.transpose(NP.array([vars()['T_'+zone][numbers-1:trainsize*len(vars()['T_'+zone])], vars()['Heatrate_'+zone][numbers-1:trainsize*len(vars()['Heatrate_'+zone])]]))
    y_test =  NP.transpose(NP.array([vars()['T_'+zone][int(0.0*len(vars()['T_'+zone])) + numbers-1:], vars()['Heatrate_'+zone][int(0.0 * len(vars()['Heatrate_'+zone])) + numbers-1:]]))

    """
    Create past values
    """

    features = x_train.shape[1]
    l_train = len(x_train)
    tmp_train = NP.zeros((l_train - numbers+1, numbers*features))

    for i in range(0,l_train-numbers+1):
        tmp_train[i,:] = x_train[i:i+numbers,0:features].reshape(1,numbers*features)
    x_train = tmp_train[:]
    x_train = NP.delete(x_train, features*(numbers-1), 1)

    l_test = len(x_test)
    tmp_test = NP.zeros((l_test - numbers+1, numbers*features))
    for i in range(0,l_test-numbers+1):
        tmp_test[i,:] = x_test[i:i+numbers,0:features].reshape(1,numbers*features)

    x_test = tmp_test[:]
    # delete the current room temperature
    x_test = NP.delete(x_test, features*(numbers-1), 1)

    """
    -------
    x_train
    -------
                    T_train(t-1) |...| SchedVal_train(t-1) | .... | P_amb_train(t-1) | SchedVal_train(t) | u_AHU_train(t)  |...| P_amb_train(t)
    sample 1            .        |...|          .          |   .  |      .           |       .           |         .       |...|       .
    .                   .        |...|          .          |   .  |      .           |       .           |         .       |...|       .
    sample n                     |...|          .          |   .  |      .           |       .           |         .       |...|       .
    """

    """
    Normalize Data
    """
    if load == 0:
        x_lb = NP.min(x_train,axis =0)
        x_ub = NP.max(x_train,axis =0)

        y_lb = NP.min(y_train,axis =0)
        y_ub = NP.max(y_train,axis =0)
    else:
        lbub = NP.load('Models\lbub_' + zone + '.npy')
        x_lb = lbub[0]
        x_ub = lbub[1]
        y_lb = lbub[2]
        y_ub = lbub[3]

    x_train = NP.divide(x_train - x_lb, x_ub - x_lb)
    y_train = NP.divide(y_train - y_lb, y_ub - y_lb)
    x_test = NP.divide(x_test - x_lb, x_ub - x_lb)
    y_test = NP.divide(y_test - y_lb, y_ub - y_lb)

    # past values are in same row - rows are shuffled here -> keeps current and
    # past values together
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
        json_file = open('Models\model_' + zone + '.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        # load weights into new model
        model.load_weights('Models\model_' + zone +'.h5')
        y_test = NP.squeeze(y_test)
        y_test = NP.multiply(y_test, y_ub-y_lb) + y_lb
        classes = NP.squeeze(model.predict(x_test))
        classes = NP.multiply(classes, y_ub-y_lb) + y_lb

        mse = NP.mean((y_test-classes)**2,axis=0)
        rmse = NP.zeros((y_train.shape[1],1))

        for i in range(y_train.shape[1]):
            rmse[i] = NP.sqrt(mse[i])
        print(zone + " RMSE: " + str(rmse[0]))
        P.subplot(2, 1, 1)
        P.plot(y_test[:,0])
        P.plot(classes[:,0])
        P.ylabel('SetPoint')
        P.legend(['ZoneTemp','NN'])
        P.title(zone)

        P.subplot(2, 1, 2)
        P.plot(vars()['u_AHU_' + zone])
        # P.show()
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
        model.add(Dense(units = y_train.shape[1], activation='linear'))

        Adam = optimizers.Adam(lr=5e-4,decay = 0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        model.compile(loss='mean_squared_error', optimizer= Adam, metrics = ['mae'])

        """
        ----------------------------------------
        Model Training
        ----------------------------------------
        """
        trained = model.fit(x_train, y_train, validation_split=0, shuffle = False, epochs=500, batch_size=1024, verbose = 2)
        if save == 1:
            model_json = model.to_json()
            with open('Models\model_' + zone + '.json', 'w') as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights('Models\model_' + zone +'.h5')
            print('Saved model to disk')
            NP.save('Models\lbub_' + zone + '.npy', [x_lb, x_ub, y_lb, y_ub])
