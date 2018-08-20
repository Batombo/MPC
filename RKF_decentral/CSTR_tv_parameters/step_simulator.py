import core_do_mpc
import data_do_mpc

from pyfmi import load_fmu
from pdb import set_trace as bp
import time
import numpy as NP
from vars import *
from casadi import *
import pylab as P


def step_simulator(model_fmu, simTime, secStep, configurations):
    secStep = secStep*60
    simTime = simTime*60

    u_mpc = NP.resize(NP.array([]),(len(configurations),6))
    tmp = NP.zeros(4)
    for i in range(len(configurations)):
        u_mpc[i,:] = configurations[i].optimizer.u_mpc

        if configurations[i].model.name == 'Coworking':
            tmp[0] = configurations[i].optimizer.u_mpc[0]
        elif configurations[i].model.name == 'MeetingNorth':
            tmp[1] = configurations[i].optimizer.u_mpc[1]
        elif configurations[i].model.name == 'MeetingSouth':
            tmp[2] = configurations[i].optimizer.u_mpc[2]
        elif configurations[i].model.name == 'Entrance':
            tmp[3] = configurations[i].optimizer.u_mpc[3]
    # Blinds
    tmp = NP.round(tmp)
    model_fmu.set('u_blinds_E', tmp[0])
    model_fmu.set('u_blinds_N', tmp[1])
    model_fmu.set('u_blinds_S', tmp[2])
    model_fmu.set('u_blinds_W', tmp[3])

    for i in range(len(configurations)):
        # AHU
        u_mpc[i,4] = NP.floor(u_mpc[i,4]*10)/10
        model_fmu.set('u_AHU_' + zones[i], u_mpc[i,4])
        # Baseboard Heaters
        model_fmu.set('u_rad_' + zones[i], u_mpc[i,5])

    # do one step simulation
    res = model_fmu.do_step(current_t=simTime, step_size=secStep, new_step=True)
    Heatrate = NP.zeros((len(configurations),1))
    for i in range(0, Heatrate.shape[0]):
        Heatrate[i,:] = model_fmu.get('Heatrate_' + zones[i])
        # Concatenate past Heatrate values with present ones
        configurations[i].simulator.Heatrate = NP.concatenate((configurations[i].simulator.Heatrate, NP.array([Heatrate[i,:]])), axis = 1)

        if u_mpc[i,4] > 0 and Heatrate[i,:] > 0:
            configurations[i].simulator.faultDetector = NP.concatenate((configurations[i].simulator.faultDetector, NP.ones((1,1)) ), axis = 1)
        else:
            configurations[i].simulator.faultDetector = NP.concatenate((configurations[i].simulator.faultDetector, NP.zeros((1,1)) ), axis = 1)
    # these states are shared by all zones
    states_all = []
    states_all.append(model_fmu.get('v_Tamb'))
    states_all.append(model_fmu.get('v_solGlobFac_E'))
    states_all.append(model_fmu.get('v_solGlobFac_N'))
    states_all.append(model_fmu.get('v_solGlobFac_S'))
    states_all.append(model_fmu.get('v_solGlobFac_W'))
    states_all.append(model_fmu.get('windspeed_Z1'))
    states_all.append(model_fmu.get('Hum_amb'))
    states_all.append(model_fmu.get('P_amb'))

    for i in range(len(configurations)):
        states = []
        states.append(model_fmu.get('T_' + zones[i]))
        blinds = [0,0,0,0]
        if zones[i] in eastSide:
            blinds[0] = tmp[0]
        if zones[i] in northSide:
            blinds[1] = tmp[1]
        if zones[i] in southSide:
            blinds[2] = tmp[2]
        if zones[i] in westSide:
            blinds[3] = tmp[3]
        for k in range(0,len(blinds)): states.append(NP.asarray([blinds[k]]))

        states.append(NP.asarray([u_mpc[i,4]]))
        states.append(NP.asarray([u_mpc[i,5]]))
        states.append(model_fmu.get('v_IG_' + zones[i]))
        for k in range(0,len(states_all)): states.append(states_all[k])
        states = NP.squeeze(states)
        
        # close loop for all configurations
        configurations[i].simulator.xf_sim = states
        # Update the initial condition for the next iteration
        configurations[i].simulator.x0_sim = configurations[i].simulator.xf_sim
        # Correction for sizes of arrays when dimension is 1
        if configurations[i].simulator.xf_sim.shape ==  ():
            configurations[i].simulator.xf_sim = NP.array([configurations[i].simulator.xf_sim])
        # Update the mpc iteration index and the time
        configurations[i].simulator.mpc_iteration = configurations[i].simulator.mpc_iteration + 1
        configurations[i].simulator.t0_sim = configurations[i].simulator.tf_sim
        configurations[i].simulator.tf_sim = configurations[i].simulator.tf_sim + configurations[i].simulator.t_step_simulator

        # only for bounds error detection
        tv_p_real = configurations[i].simulator.tv_p_real_now(configurations[i].simulator.t0_sim)
        rhs_unscaled = substitute(configurations[i].model.rhs, configurations[i].model.x, configurations[i].model.x * configurations[i].model.ocp.x_scaling)/configurations[i].model.ocp.x_scaling
        rhs_unscaled = substitute(rhs_unscaled, configurations[i].model.u, configurations[i].model.u * configurations[i].model.ocp.u_scaling)
        rhs_fcn = Function('rhs_fcn',[configurations[i].model.x,vertcat(configurations[i].model.u,configurations[i].model.tv_p)],[rhs_unscaled])
        x_next = rhs_fcn(configurations[i].simulator.x0_sim,vertcat(u_mpc[i,:],tv_p_real))

        x_lb = NP.squeeze(NP.asarray(configurations[i].model.ocp.x_lb))
        x_ub = NP.squeeze(NP.asarray(configurations[i].model.ocp.x_ub))
        if (NP.squeeze(NP.asarray(x_next)) - x_lb < 0).all() or (x_ub - NP.squeeze(NP.asarray(x_next))  < 0).all():
            bp()


def f_const(var):
    if var <= 0*100:
        return 17
    elif var <= 1*100:
        return 22
    elif var <= 2*100:
        return 16
    elif var <= 3*100:
        return 21
    elif var <= 4*100:
        return 19
    elif var <= 5*100:
        return 20
    elif var <= 6*100:
        return 22
    elif var <= 7*100:
        return 18
    elif var <= 8*100:
        return 22
    else:
        return 20

def f_const_ahu(var):
    if var <= 0*100:
        return 0.6
    elif var <= 1*100:
        return 0.1
    elif var <= 2*100:
        return 0.8
    elif var <= 3*100:
        return 0.3
    elif var <= 4*100:
        return 0.4
    elif var <= 5*100:
        return 0.7
    elif var <= 6*100:
        return 0
    elif var <= 7*100:
        return 0.5
    elif var <= 8*100:
        return 0.2
    else:
        return 0.9

def compare(model_fmu, simTime, secStep, configurations):
    secStep = secStep*60
    simTime = simTime*60
    duration = 144*10
    sched = NP.zeros((duration,6))

    for i in range(len(configurations)):
        vars()['result_Ep_' + zones[i]] = NP.zeros((duration,configurations[i].model.x.shape[0]))
        vars()['result_NN_' + zones[i]] = NP.zeros((duration,configurations[i].model.x.shape[0]))

        for j in range(0,duration):
            u_mpc = configurations[i].optimizer.u_mpc
            u_mpc = NP.asarray(u_mpc, dtype=NP.float32)
            u_mpc[5] = float(f_const(j)) #np.round(np.random.normal(17,0.1,1),2)
            u_mpc[1] = 1
            tv_p_real = configurations[i].simulator.tv_p_real_now(configurations[i].simulator.t0_sim)
            rhs_unscaled = substitute(configurations[i].model.rhs, configurations[i].model.x, configurations[i].model.x * configurations[i].model.ocp.x_scaling)/configurations[i].model.ocp.x_scaling
            rhs_unscaled = substitute(rhs_unscaled, configurations[i].model.u, configurations[i].model.u * configurations[i].model.ocp.u_scaling)
            rhs_fcn = Function('rhs_fcn',[configurations[i].model.x,vertcat(configurations[i].model.u,configurations[i].model.tv_p)],[rhs_unscaled])
            x_next = rhs_fcn(configurations[i].simulator.x0_sim,vertcat(u_mpc,tv_p_real))
            configurations[i].simulator.xf_sim = NP.squeeze(NP.array(x_next))
            configurations[i].simulator.x0_sim = configurations[i].simulator.xf_sim

            configurations[i].simulator.mpc_iteration = configurations[i].simulator.mpc_iteration + 1
            configurations[i].simulator.t0_sim = configurations[i].simulator.tf_sim
            configurations[i].simulator.tf_sim = configurations[i].simulator.tf_sim + configurations[i].simulator.t_step_simulator

            vars()['result_NN_' + zones[i]][j,:] = configurations[i].simulator.xf_sim
    for j in range(0,duration):
        u_mpc = configurations[0].optimizer.u_mpc
        u_mpc = NP.asarray(u_mpc, dtype=NP.float32)
        u_mpc[5] = float(f_const(j)) #np.round(np.random.normal(17,0.1,1),2)
        u_mpc[1] = 1
        """
        Blinds
        """
        model_fmu.set('u_blinds_E', u_mpc[0])
        model_fmu.set('u_blinds_N', u_mpc[1])
        model_fmu.set('u_blinds_S', u_mpc[2])
        model_fmu.set('u_blinds_W', u_mpc[3])

        for i in range(len(configurations)):
            """
            AHU
            """
            model_fmu.set('u_AHU_' + zones[i], u_mpc[4])
            """
            Baseboard Heater
            """
            model_fmu.set('u_rad_' + zones[i], u_mpc[5])
        # do one step simulation
        res = model_fmu.do_step(current_t=simTime, step_size=secStep, new_step=True)

        states_all = []
        states_all.append(model_fmu.get('u_blinds_E_val'))
        states_all.append(model_fmu.get('u_blinds_N_val'))
        states_all.append(model_fmu.get('u_blinds_S_val'))
        states_all.append(model_fmu.get('u_blinds_W_val'))

        states_all.append(model_fmu.get('v_Tamb'))
        states_all.append(model_fmu.get('v_solGlobFac_E'))
        states_all.append(model_fmu.get('v_solGlobFac_N'))
        states_all.append(model_fmu.get('v_solGlobFac_S'))
        states_all.append(model_fmu.get('v_solGlobFac_W'))
        states_all.append(model_fmu.get('windspeed_Z1'))
        states_all.append(model_fmu.get('Hum_amb'))
        states_all.append(model_fmu.get('P_amb'))

        for i in range(len(configurations)):
            states = []
            states.append(model_fmu.get('T_' + zones[i]))
            for k in range(0,4): states.append(states_all[k])
            states.append(NP.asarray([u_mpc[4]]))
            states.append(NP.asarray([u_mpc[5]]))
            states.append(model_fmu.get('v_IG_' + zones[i]))
            for k in range(4,len(states_all)): states.append(states_all[k])
            states = NP.squeeze(states)
            vars()['result_Ep_' + zones[i]][j,:] = states

        simTime += secStep
        sched[j,:] = u_mpc
    for i in range(len(configurations)):
        result_Ep = vars()['result_Ep_' + zones[i]]
        result_NN = vars()['result_NN_' + zones[i]]
        rmse = NP.sqrt(NP.mean((result_Ep[:,0]-result_NN[:,0])**2,axis=0))
        print("RMSE: " + str(rmse))
        # NP.argmax(NP.abs(result_NN[:,0]-result_Ep[:,0]),axis=0)
        P.subplot(3, 1, 1)
        P.plot(result_NN[:,0], linewidth = 2.0)
        P.plot(result_Ep[:,0], linewidth = 2.0)
        P.plot(sched[:,-1])
        P.ylabel('ZoneTemp')
        P.legend(['NN','E+','Setpoint'])
        P.title(zones[i])
        P.subplot(3, 1, 2)
        P.plot(result_NN[:,0] - result_Ep[:,0])
        # P.plot(result_NN[:,8] - result_Ep[:,8])
        P.ylabel('Difference')

        P.subplot(3, 1, 3)
        P.plot(result_NN[:,7])
        P.plot(result_Ep[:,7])
        P.ylabel('Difference v_IG_Offices')
        P.show()

        P.subplot(6,1,1)
        P.plot(result_NN[:,7])
        P.plot(result_Ep[:,7])

        P.subplot(6,1,2)
        P.plot(result_NN[:,8])
        P.plot(result_Ep[:,8])
        P.subplot(6,1,3)
        P.plot(result_NN[:,9])
        P.plot(result_Ep[:,9])
        P.subplot(6,1,4)
        P.plot(result_NN[:,13])
        P.plot(result_Ep[:,13])
        P.subplot(6,1,5)
        P.plot(result_NN[:,14])
        P.plot(result_Ep[:,14])
        P.subplot(6,1,6)
        P.plot(result_NN[:,15])
        P.plot(result_Ep[:,15])

        P.show()


        # P.savefig('fig_' + zone)
