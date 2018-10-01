from pdb import set_trace as bp
from pyfmi import load_fmu
import numpy as np
import pylab as P
from random import randint
import random

from tempfile import TemporaryFile


days = 365
hours = 24
minutes = 60
seconds = 60
EPTimeStep = 6 #Number of timesteps per hour in EnergyPlus
dayduration = hours*minutes*seconds

numSteps = days*hours*EPTimeStep
timeStop = days*hours*minutes*seconds
secStep = timeStop/numSteps

# data =  np.load('full_reg.npy')
# np.save('disturbances.npy', [data[4,0:6*24*365], data[3,0:6*24*365], data[5,0:6*24*365], data[6,0:6*24*365], data[7,0:6*24*365], data[8,0:6*24*365]])


# Setup input function
def f_gauss(time, rand =0):
    if time % 1 == 0 or rand == 1:
        mu, sigma = 20, 10
        s = np.round(np.random.normal(mu,sigma,1),2)
        return s
    else:
        return u_AHU1_noERC[-1]

def f_const_gauss(time, min, max, rand = 0):
    if time % 1 == 0 or rand == 1:
        if time <= 7 or time >= 19:
            mu, sigma = min, 0.1
        else:
            mu, sigma = max, 0.1
        s = np.round(np.random.normal(mu,sigma,1),2)
        if s < 14:
            s = 14
        elif s > 25:
            s = 25
        return s
    else:
        return SchedVal[-1]
def f_const(time, min, max):
    if time <= 7 or time >= 19:
        return min
    else:
        return max

def f_stairs(sched,time, min, max, stepsize, default):
    eps = 1e-5
    if time <= 600:
        return default
    elif time > 600 and sched[-1] < max and np.array(sched[-1]) > np.array(sched[-2]) - eps : #  time % 1 == 0 and
        return np.array(sched[-1]) + stepsize
    elif time > 600*7 and sched[-1] > min and np.array(sched[-1]) < np.array(sched[-2]) + eps : # time % 1 == 0 and
        return np.array(sched[-1]) - stepsize
    else:
        return sched[-1]

def simTime2Clock(simTime, days):
    return simTime/600 * float(1)/EPTimeStep - days * dayduration/600 *  float(1)/EPTimeStep

def radiation2shading(vsol, threshold):
    if vsol[-1] < threshold:
        return 0
    else:
        return 1
#7
for i in range(0,1):
    # Load FMU created from compile_fmu() or EnergyPlusToFMU
    if i == 0:
        modelName = 'RKF_Berlin'
    elif i == 1:
        modelName = 'RKF_Chemnitz'
    elif i == 2:
        modelName = 'RKF_Gorzow'
    elif i == 3:
        modelName = 'RKF_Hamburg'
    elif i == 4:
        modelName = 'RKF_Poznan'
    elif i == 5:
        modelName = 'RKF_Slubice'
    else:
        modelName = 'RKF_Zielona'
    model = load_fmu(modelName+'.fmu')
    if i == 0:
        SchedVal = []
        HeatRate = []
        ZoneTemp = []

        v_Tamb = []
        v_IG_Offices = []
        v_solGlobFac_E = []
        v_solGlobFac_N = []
        v_solGlobFac_S = []
        v_solGlobFac_W = []
        v_windspeed = []

        # Hum_zone = []
        Hum_amb = []
        P_amb = []

        u_blinds_E = []
        u_blinds_N = []
        u_blinds_S = []
        u_blinds_W = []

        u_AHU1_noERC = []


    # Setup simulation parameters
    index = 0
    simTime = 86400*0
    timeStop = days*hours*minutes*seconds
    timeStop+= simTime
    sTime = 0
    # Load options
    opts = model.simulate_options()
    opts['ncp'] = numSteps
    # Manually initialize
    opts['initialize'] = False

    model.initialize(simTime,timeStop)
    days = 0

    while simTime < timeStop:
        clock = simTime2Clock(sTime, days)
        if i == 0:
            setp = f_const(clock,17,20)
            model.set('u_rad_OfficesZ1',setp)
            model.set('u_AHU1_noERC', 0)
            model.set('u_blinds_N', 0)
            model.set('u_blinds_W', 0)
        elif i == 1:
            model.set('u_rad_OfficesZ1',18)
            model.set('u_AHU1_noERC', 1)
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 50))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 50))
        elif i == 2:
            setp = f_const(clock,18,20)
            model.set('u_rad_OfficesZ1',setp)
            model.set('u_AHU1_noERC', 0.5)
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 100))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 100))
        elif i == 3:
            setp = f_const(clock,16,18)
            model.set('u_rad_OfficesZ1',setp)
            model.set('u_AHU1_noERC', 0.7)
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 250))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 250))
        elif i == 4:
            setp = f_const(clock,17,21)
            model.set('u_rad_OfficesZ1',setp)
            model.set('u_AHU1_noERC',0)
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 500))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 500))
        elif i == 5:
            model.set('u_rad_OfficesZ1',f_stairs(SchedVal, simTime, 16, 23, 0.1, 16))
            model.set('u_AHU1_noERC',f_stairs(u_AHU1_noERC, simTime, 0, 1, 0.01, 0))
            model.set('u_blinds_N', random.random())
            model.set('u_blinds_W', random.random())

        else:
            setp = f_const(clock,19,17)
            model.set('u_rad_OfficesZ1',setp)
            model.set('u_AHU1_noERC', 0)
            model.set('u_blinds_N', f_stairs(u_blinds_N, simTime, 0, 1, 0.01, 0))
            model.set('u_blinds_W', f_stairs(u_blinds_W, simTime, 0, 1, 0.01, 0))




        res = model.do_step(current_t=simTime, step_size=secStep, new_step=True)
        SchedVal.append(model.get('SchedVal_Z1'))
        ZoneTemp.append(model.get('Tzone_1'))
        v_Tamb.append(model.get('v_Tamb'))
        v_IG_Offices.append(model.get('v_IG_Offices'))
        v_solGlobFac_E.append(model.get('v_solGlobFac_E'))
        v_solGlobFac_N.append(model.get('v_solGlobFac_N'))
        v_solGlobFac_S.append(model.get('v_solGlobFac_S'))
        v_solGlobFac_W.append(model.get('v_solGlobFac_W'))
        u_AHU1_noERC.append(model.get('u_AHU1_noERC_SchedVal'))
        HeatRate.append(model.get('HeatRate_Z1'))
        v_windspeed.append(model.get('windspeed_Z1'))


        Hum_amb.append(model.get('Hum_amb'))
        # Hum_zone.append(model.get('Hum_zone'))
        P_amb.append(model.get('P_amb'))

        if i == 6:
            u_blinds_N.append(f_stairs(u_blinds_N, simTime, 0, 1, 0.01, 0))
            u_blinds_W.append(f_stairs(u_blinds_W, simTime, 0, 1, 0.01, 0))
        elif i == 5:
            u_blinds_N.append(random.random())
            u_blinds_W.append(random.random())
        else:
            u_blinds_N.append(model.get('u_blinds_N_val'))
            u_blinds_W.append(model.get('u_blinds_W_val'))


        simTime += secStep
        sTime += secStep
        index += 1
        if simTime % dayduration == 0:
            days += 1
    print "index :" + str(i)

    # t = np.linspace(0.0,timeStop,numSteps)
    # fig = P.figure(1)
    # P.clf()
    # P.subplot(3, 1, 1)
    # P.plot(t/(3600*24), v_Tamb[i*6*24*days:])
    # P.ylabel('Temperature ' + u'\u2103')
    # P.xlabel('Time (days)')
    # P.subplot(3, 1, 2)
    # P.plot(t/(3600*24), ZoneTemp[i*6*24*days:])
    # P.plot(t/(3600*24), SchedVal[i*6*24*days:])
    # P.ylabel('Temperature ' + u'\u2103')
    # P.xlabel('Time (days)')
    # P.legend(['ZoneTemp', 'SchedVal'])
    # P.subplot(3, 1, 3)
    # P.plot(t/(3600*24), HeatRate[i*6*24*days:])
    # P.xlabel('Time (days)')
    # P.show()
    model = None
np.save('test_data.npy', [SchedVal, HeatRate, u_AHU1_noERC, ZoneTemp, v_Tamb, v_IG_Offices,
v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W,
u_blinds_N, u_blinds_W,
v_windspeed, Hum_amb, P_amb])
# np.save('disturbances.npy', [v_IG_Offices, v_Tamb, v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W, v_windspeed, Hum_amb, P_amb])
