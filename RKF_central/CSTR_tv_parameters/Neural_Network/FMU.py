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
        modelName = 'RKF_Potsdam'
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
        SchedVal_Coworking = []
        SchedVal_Corridor = []
        SchedVal_Entrance = []
        SchedVal_LabNorth = []
        SchedVal_LabSouth = []
        SchedVal_MeetingSouth = []
        SchedVal_MeetingNorth = []
        SchedVal_Nerdroom1 = []
        SchedVal_Nerdroom2 = []
        SchedVal_RestroomM = []
        SchedVal_RestroomW = []
        SchedVal_Space01 = []
        SchedVal_Stairway = []


        Heatrate_Coworking = []
        Heatrate_Corridor = []
        Heatrate_Entrance = []
        Heatrate_LabNorth = []
        Heatrate_LabSouth = []
        Heatrate_MeetingSouth = []
        Heatrate_MeetingNorth = []
        Heatrate_Nerdroom1 = []
        Heatrate_Nerdroom2 = []
        Heatrate_RestroomM = []
        Heatrate_RestroomW = []
        Heatrate_Space01 = []
        Heatrate_Stairway = []

        T_Coworking = []
        T_Corridor = []
        T_Elevator = []
        T_Entrance = []
        T_LabNorth = []
        T_LabSouth = []
        T_MeetingSouth = []
        T_MeetingNorth = []
        T_Nerdroom1 = []
        T_Nerdroom2 = []
        T_RestroomM = []
        T_RestroomW = []
        T_Space01 = []
        T_Stairway = []

        v_IG_Coworking = []
        v_IG_Corridor = []
        v_IG_Elevator = []
        v_IG_Entrance = []
        v_IG_LabNorth = []
        v_IG_LabSouth = []
        v_IG_MeetingSouth = []
        v_IG_MeetingNorth = []
        v_IG_Nerdroom1 = []
        v_IG_Nerdroom2 = []
        v_IG_RestroomM = []
        v_IG_RestroomW = []
        v_IG_Space01 = []
        v_IG_Stairway = []

        v_Tamb = []
        v_solGlobFac_E = []
        v_solGlobFac_N = []
        v_solGlobFac_S = []
        v_solGlobFac_W = []
        v_windspeed = []

        Hum_amb = []
        P_amb = []

        u_blinds_E = []
        u_blinds_N = []
        u_blinds_S = []
        u_blinds_W = []

        u_AHU_Coworking = []
        u_AHU_Corridor = []
        u_AHU_Entrance = []
        u_AHU_LabNorth = []
        u_AHU_LabSouth = []
        u_AHU_MeetingSouth = []
        u_AHU_MeetingNorth = []
        u_AHU_Nerdroom1 = []
        u_AHU_Nerdroom2 = []
        u_AHU_RestroomW = []
        u_AHU_Space01 = []
        u_AHU_Stairway = []


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
            setp_urad = f_const(clock,20,22)
            setp_AHU = 0

            model.set('u_blinds_E', 0)
            model.set('u_blinds_N', 0)
            model.set('u_blinds_S', 0)
            model.set('u_blinds_W', 0)
        elif i == 1:
            setp_urad = 18
            setp_AHU = 1

            model.set('u_blinds_E', radiation2shading(v_solGlobFac_E, 50))
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 50))
            model.set('u_blinds_S', radiation2shading(v_solGlobFac_S, 50))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 50))
        elif i == 2:
            setp_urad = f_const(clock,18,20)
            setp_AHU = 0.5

            model.set('u_blinds_E', radiation2shading(v_solGlobFac_E, 100))
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 100))
            model.set('u_blinds_S', radiation2shading(v_solGlobFac_S, 100))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 100))
        elif i == 3:
            setp_urad = f_const(clock,16,18)
            setp_AHU = 0.7

            model.set('u_blinds_E', radiation2shading(v_solGlobFac_E, 250))
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 250))
            model.set('u_blinds_S', radiation2shading(v_solGlobFac_S, 250))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 250))
        elif i == 4:
            setp_urad = f_const(clock,17,21)
            setp_AHU = 0

            model.set('u_blinds_E', radiation2shading(v_solGlobFac_E, 500))
            model.set('u_blinds_N', radiation2shading(v_solGlobFac_N, 500))
            model.set('u_blinds_S', radiation2shading(v_solGlobFac_S, 500))
            model.set('u_blinds_W', radiation2shading(v_solGlobFac_W, 500))
        elif i == 5:
            setp_urad = f_stairs(SchedVal_Coworking, simTime, 16, 22, 0.1, 16)
            setp_AHU = f_stairs(u_AHU_Coworking, simTime, 0, 1, 0.01, 0)

            model.set('u_blinds_E', random.random())
            model.set('u_blinds_N', random.random())
            model.set('u_blinds_S', random.random())
            model.set('u_blinds_W', random.random())

        else:
            setp_urad = f_const(clock,19,17)
            setp_AHU = 0

            model.set('u_blinds_E', f_stairs(u_blinds_E, simTime, 0, 1, 0.01, 0))
            model.set('u_blinds_N', f_stairs(u_blinds_N, simTime, 0, 1, 0.01, 0))
            model.set('u_blinds_S', f_stairs(u_blinds_S, simTime, 0, 1, 0.01, 0))
            model.set('u_blinds_W', f_stairs(u_blinds_W, simTime, 0, 1, 0.01, 0))


        model.set('u_rad_Coworking',setp_urad)
        model.set('u_rad_Corridor',setp_urad)
        model.set('u_rad_Entrance',setp_urad)
        model.set('u_rad_LabNorth',setp_urad)
        model.set('u_rad_LabSouth',setp_urad)
        model.set('u_rad_MeetingSouth',setp_urad)
        model.set('u_rad_MeetingNorth',setp_urad)
        model.set('u_rad_Nerdroom1',setp_urad)
        model.set('u_rad_Nerdroom2',setp_urad)
        model.set('u_rad_RestroomM',setp_urad)
        model.set('u_rad_RestroomW',setp_urad)
        model.set('u_rad_Space01',setp_urad)
        model.set('u_rad_Stairway',setp_urad)


        model.set('u_AHU_Coworking', setp_AHU)
        model.set('u_AHU_Corridor', setp_AHU)
        model.set('u_AHU_Entrance', setp_AHU)
        model.set('u_AHU_LabNorth', setp_AHU)
        model.set('u_AHU_LabSouth', setp_AHU)
        model.set('u_AHU_MeetingSouth', setp_AHU)
        model.set('u_AHU_MeetingNorth', setp_AHU)
        model.set('u_AHU_Nerdroom1', setp_AHU)
        model.set('u_AHU_Nerdroom2', setp_AHU)
        model.set('u_AHU_RestroomW', setp_AHU)
        model.set('u_AHU_Space01', setp_AHU)
        model.set('u_AHU_Stairway', setp_AHU)

        res = model.do_step(current_t=simTime, step_size=secStep, new_step=True)

        SchedVal_Coworking.append(setp_urad)
        SchedVal_Corridor.append(setp_urad)
        SchedVal_Entrance.append(setp_urad)
        SchedVal_LabNorth.append(setp_urad)
        SchedVal_LabSouth.append(setp_urad)
        SchedVal_MeetingSouth.append(setp_urad)
        SchedVal_MeetingNorth.append(setp_urad)
        SchedVal_Nerdroom1.append(setp_urad)
        SchedVal_Nerdroom2.append(setp_urad)
        SchedVal_RestroomM.append(setp_urad)
        SchedVal_RestroomW.append(setp_urad)
        SchedVal_Space01.append(setp_urad)
        SchedVal_Stairway.append(setp_urad)



        T_Coworking.append(model.get('T_Coworking'))
        T_Corridor.append(model.get('T_Corridor'))
        T_Elevator.append(model.get('T_Elevator'))
        T_Entrance.append(model.get('T_Entrance'))
        T_LabNorth.append(model.get('T_LabNorth'))
        T_LabSouth.append(model.get('T_LabSouth'))
        T_MeetingSouth.append(model.get('T_MeetingSouth'))
        T_MeetingNorth.append(model.get('T_MeetingNorth'))
        T_Nerdroom1.append(model.get('T_Nerdroom1'))
        T_Nerdroom2.append(model.get('T_Nerdroom2'))
        T_RestroomM.append(model.get('T_RestroomM'))
        T_RestroomW.append(model.get('T_RestroomW'))
        T_Space01.append(model.get('T_Space01'))
        T_Stairway.append(model.get('T_Stairway'))



        v_Tamb.append(model.get('v_Tamb'))
        v_solGlobFac_E.append(model.get('v_solGlobFac_E'))
        v_solGlobFac_N.append(model.get('v_solGlobFac_N'))
        v_solGlobFac_S.append(model.get('v_solGlobFac_S'))
        v_solGlobFac_W.append(model.get('v_solGlobFac_W'))

        v_IG_Coworking.append(model.get('v_IG_Coworking'))
        v_IG_Corridor.append(model.get('v_IG_Corridor'))
        v_IG_Elevator.append(model.get('v_IG_Elevator'))
        v_IG_Entrance.append(model.get('v_IG_Entrance'))
        v_IG_LabNorth.append(model.get('v_IG_LabNorth'))
        v_IG_LabSouth.append(model.get('v_IG_LabSouth'))
        v_IG_MeetingSouth.append(model.get('v_IG_MeetingSouth'))
        v_IG_MeetingNorth.append(model.get('v_IG_MeetingNorth'))
        v_IG_Nerdroom1.append(model.get('v_IG_Nerdroom1'))
        v_IG_Nerdroom2.append(model.get('v_IG_Nerdroom2'))
        v_IG_RestroomM.append(model.get('v_IG_RestroomM'))
        v_IG_RestroomW.append(model.get('v_IG_RestroomW'))
        v_IG_Space01.append(model.get('v_IG_Space01'))
        v_IG_Stairway.append(model.get('v_IG_Stairway'))





        u_AHU_Coworking.append(setp_AHU)
        u_AHU_Corridor.append(setp_AHU)
        u_AHU_Entrance.append(setp_AHU)
        u_AHU_LabNorth.append(setp_AHU)
        u_AHU_LabSouth.append(setp_AHU)
        u_AHU_MeetingSouth.append(setp_AHU)
        u_AHU_MeetingNorth.append(setp_AHU)
        u_AHU_Nerdroom1.append(setp_AHU)
        u_AHU_Nerdroom2.append(setp_AHU)
        u_AHU_RestroomW.append(setp_AHU)
        u_AHU_Space01.append(setp_AHU)
        u_AHU_Stairway.append(setp_AHU)

        Heatrate_Coworking.append(model.get('Heatrate_Coworking'))
        Heatrate_Corridor.append(model.get('Heatrate_Corridor'))
        Heatrate_Entrance.append(model.get('Heatrate_Entrance'))
        Heatrate_LabNorth.append(model.get('Heatrate_LabNorth'))
        Heatrate_LabSouth.append(model.get('Heatrate_LabSouth'))
        Heatrate_MeetingSouth.append(model.get('Heatrate_MeetingSouth'))
        Heatrate_MeetingNorth.append(model.get('Heatrate_MeetingNorth'))
        Heatrate_Nerdroom1.append(model.get('Heatrate_Nerdroom1'))
        Heatrate_Nerdroom2.append(model.get('Heatrate_Nerdroom2'))
        Heatrate_RestroomM.append(model.get('Heatrate_RestroomM'))
        Heatrate_RestroomW.append(model.get('Heatrate_RestroomW'))
        Heatrate_Space01.append(model.get('Heatrate_Space01'))
        Heatrate_Stairway.append(model.get('Heatrate_Stairway'))

        v_windspeed.append(model.get('windspeed_Z1'))

        Hum_amb.append(model.get('Hum_amb'))
        P_amb.append(model.get('P_amb'))

        if i == 5:
            u_blinds_E.append(random.random())
            u_blinds_N.append(random.random())
            u_blinds_S.append(random.random())
            u_blinds_W.append(random.random())
        elif i == 6:
            u_blinds_E.append(f_stairs(u_blinds_E, simTime, 0, 1, 0.01, 0))
            u_blinds_N.append(f_stairs(u_blinds_N, simTime, 0, 1, 0.01, 0))
            u_blinds_S.append(f_stairs(u_blinds_S, simTime, 0, 1, 0.01, 0))
            u_blinds_W.append(f_stairs(u_blinds_W, simTime, 0, 1, 0.01, 0))
        else:
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

    # t = np.linspace(0.0,timeStop,numSteps)
    # fig = P.figure(1)
    # P.clf()
    # P.subplot(3, 1, 1)
    # P.plot(t/(3600*24), u_AHU_Corridor[i*6*24*days:])
    # P.ylabel('Temperature ' + u'\u2103')
    # P.xlabel('Time (days)')
    # P.subplot(3, 1, 2)
    # P.plot(t/(3600*24), T_Corridor[i*6*24*days:])
    # P.plot(t/(3600*24),  SchedVal_Corridor[i*6*24*days:])
    # P.ylabel('Temperature ' + u'\u2103')
    # P.xlabel('Time (days)')
    # P.legend(['ZoneTemp', 'SchedVal'])
    # P.subplot(3, 1, 3)
    # P.plot(t/(3600*24), Heatrate_Corridor[i*6*24*days:])
    # P.xlabel('Time (days)')
    # P.show()
    model = None
# np.save('full.npy', [
#         SchedVal_Coworking,
#         SchedVal_Corridor,
#         SchedVal_Entrance,
#         SchedVal_LabNorth,
#         SchedVal_LabSouth,
#         SchedVal_MeetingSouth,
#         SchedVal_MeetingNorth,
#         SchedVal_Nerdroom1,
#         SchedVal_Nerdroom2,
#         SchedVal_RestroomM,
#         SchedVal_RestroomW,
#         SchedVal_Space01,
#         SchedVal_Stairway,

#         Heatrate_Coworking,
#         Heatrate_Corridor,
#         Heatrate_Entrance,
#         Heatrate_LabNorth,
#         Heatrate_LabSouth,
#         Heatrate_MeetingSouth,
#         Heatrate_MeetingNorth,
#         Heatrate_Nerdroom1,
#         Heatrate_Nerdroom2,
#         Heatrate_RestroomM,
#         Heatrate_RestroomW,
#         Heatrate_Space01,
#         Heatrate_Stairway,
#
#
#         u_AHU_Coworking,
#         u_AHU_Corridor,
#         u_AHU_Entrance,
#         u_AHU_LabNorth,
#         u_AHU_LabSouth,
#         u_AHU_MeetingSouth,
#         u_AHU_MeetingNorth,
#         u_AHU_Nerdroom1,
#         u_AHU_Nerdroom2,
#         u_AHU_RestroomW,
#         u_AHU_Space01,
#         u_AHU_Stairway,
#
#
#         T_Coworking,
#         T_Corridor,
#         T_Elevator,
#         T_Entrance,
#         T_LabNorth,
#         T_LabSouth,
#         T_MeetingSouth,
#         T_MeetingNorth,
#         T_Nerdroom1,
#         T_Nerdroom2,
#         T_RestroomM,
#         T_RestroomW,
#         T_Space01,
#         T_Stairway,
#
#         v_Tamb,
#
#         v_IG_Coworking,
#         v_IG_Corridor,
#         v_IG_Elevator,
#         v_IG_Entrance,
#         v_IG_LabNorth,
#         v_IG_LabSouth,
#         v_IG_MeetingSouth,
#         v_IG_MeetingNorth,
#         v_IG_Nerdroom1,
#         v_IG_Nerdroom2,
#         v_IG_RestroomM,
#         v_IG_RestroomW,
#         v_IG_Space01,
#         v_IG_Stairway,
#
#         v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W,
#
#         u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W,
#         v_windspeed, Hum_amb, P_amb])
np.save('disturbances.npy', [
        v_IG_Coworking,
        v_IG_Corridor,
        v_IG_Elevator,
        v_IG_Entrance,
        v_IG_LabNorth,
        v_IG_LabSouth,
        v_IG_MeetingSouth,
        v_IG_MeetingNorth,
        v_IG_Nerdroom1,
        v_IG_Nerdroom2,
        v_IG_RestroomM,
        v_IG_RestroomW,
        v_IG_Space01,
        v_IG_Stairway,

        v_Tamb,

        v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W,
        v_windspeed, Hum_amb, P_amb])
