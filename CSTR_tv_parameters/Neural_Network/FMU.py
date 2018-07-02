from pdb import set_trace as bp
from pyfmi import load_fmu
import numpy as np
import pylab as P
from random import randint

from tempfile import TemporaryFile

modelName = 'Model'
days = 59
hours = 24
minutes = 60
seconds = 60
EPTimeStep = 6 #Number of timesteps per hour in EnergyPlus
dayduration = hours*minutes*seconds

numSteps = days*hours*EPTimeStep
timeStop = days*hours*minutes*seconds
secStep = timeStop/numSteps

# Setup input function
def f_gauss(time):
    if time % 1 == 0:
        # return randint(14, 25)
        mu, sigma = 20, 3
        s = np.round(np.random.normal(mu,sigma,1),0)
        if s < 14:
            s = 14
        elif s > 25:
            s = 25
        return s
    else:
        return SchedVal[-1]

def f_const_gauss(time, min, max, rand = 0):
    if time % 1 == 0 or rand == 1:
        if time <= 7 or time >= 19:
            mu, sigma = min, 0.75
        else:
            mu, sigma = max, 0.75
        s = np.round(np.random.normal(mu,sigma,1),1)
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

def f_stairs(time):
    eps = 1e-5
    if time <= 600:
        return 20
    elif time > 600 and SchedVal[-1] < 25 and np.array(SchedVal[-1]) > np.array(SchedVal[-2]) - eps : #  time % 1 == 0 and
        return np.array(SchedVal[-1]) + 0.1
    elif time > 600*7 and SchedVal[-1] > 20 and np.array(SchedVal[-1]) < np.array(SchedVal[-2]) + eps : # time % 1 == 0 and
        return np.array(SchedVal[-1]) - 0.1
    else:
        return SchedVal[-1]

def simTime2Clock(simTime, days):
    return simTime/600 * float(1)/EPTimeStep - days * dayduration/600 *  float(1)/EPTimeStep

for i in range(0,8):
    # Load FMU created from compile_fmu() or EnergyPlusToFMU
    model = load_fmu(modelName+'.fmu')#, log_level=5)
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

        u_blinds_E = []
        u_blinds_N = []
        u_blinds_S = []
        u_blinds_W = []
        # u_AHU1_noERC = []
        # v_Tgnd = []


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
    # Simulate over the course of a year at timestep equal to EnergyPlus
    while simTime < timeStop:
        clock = simTime2Clock(sTime, days)
        if i == 0:
            model.set('u_rad_OfficesZ1',f_const_gauss(clock,15,22,1))
        elif i == 1:
            model.set('u_rad_OfficesZ1',f_stairs(simTime))
        elif i == 2:
            model.set('u_rad_OfficesZ1',f_const(clock,17,22))
        elif i == 3:
            model.set('u_rad_OfficesZ1',f_const(clock,18,23))
        elif i == 4:
            model.set('u_rad_OfficesZ1',f_const(clock,16,21))
        elif i == 5:
            model.set('u_rad_OfficesZ1',f_const_gauss(clock,15,22,1))
        elif i == 6:
            model.set('u_rad_OfficesZ1',f_const_gauss(clock,17,23,1))
        else:
            model.set('u_rad_OfficesZ1',f_const(clock,15,20))

        model.set('u_AHU1_noERC', 0)

        res = model.do_step(current_t=simTime, step_size=secStep, new_step=True)
        SchedVal.append(model.get('SchedVal_Z1'))
        HeatRate.append(model.get('HeatRate_Z1'))
        ZoneTemp.append(model.get('Tzone_1'))
        v_Tamb.append(model.get('v_Tamb'))
        v_IG_Offices.append(model.get('v_IG_Offices'))
        v_solGlobFac_E.append(model.get('v_solGlobFac_E'))
        v_solGlobFac_N.append(model.get('v_solGlobFac_N'))
        v_solGlobFac_S.append(model.get('v_solGlobFac_S'))
        v_solGlobFac_W.append(model.get('v_solGlobFac_W'))
        u_blinds_E.append(model.get('u_blinds_E_val'))
        u_blinds_N.append(model.get('u_blinds_N_val'))
        u_blinds_S.append(model.get('u_blinds_S_val'))
        u_blinds_W.append(model.get('u_blinds_W_val'))

        simTime += secStep
        sTime += secStep
        index += 1
        if simTime % dayduration == 0:
            days += 1
    print "index :" + str(i)
    t = np.linspace(0.0,timeStop,numSteps)
    # fig = P.figure(1)
    # P.clf()
    # P.plot(t/(3600*24), SchedVal)
    # P.plot(t/(3600*24), ZoneTemp)
    # P.ylabel('Temperature ' + u'\u2103')
    # P.xlabel('Time (days)')
    # P.legend(['SchedVal','ZoneTemp'])
    # P.show()
    #
    # fig = P.figure(2)
    # P.clf()
    # P.plot(t/(3600*24), HeatRate)
    # P.ylabel('HeatRate ' + 'W')
    # P.xlabel('Time (days)')
    # P.show()
    model = None
np.save('2_reg.npy', [SchedVal, HeatRate, ZoneTemp, v_Tamb, v_IG_Offices, v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W])
# np.save('disturbances.npy', [v_IG_Offices, v_Tamb, v_Tgnd, v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W])
