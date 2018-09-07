#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2016 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.
#

from casadi import *
import numpy as NP
import core_do_mpc
import scipy.io
from keras.models import model_from_json
from pdb import set_trace as bp
from vars import *
def model(zone):
    lbub = NP.load('Neural_Network\Models\year\lbub_'+zone+'.npy')
    x_lb_NN = lbub[0]
    x_ub_NN = lbub[1]
    y_lb_NN = lbub[2]
    y_ub_NN = lbub[3]

    json_file = open('Neural_Network\Models\year\model_'+zone+'.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights('Neural_Network\Models\year\model_'+zone+'.h5')
    Theta = {}
    i = 0
    for j in range(len(model.layers)):
        weights = model.layers[j].get_weights()
        Theta['Theta'+str(i+1)] =  NP.insert(weights[0].transpose(),0,weights[1].transpose(),axis=1)
        i+=1

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    alpha   = MX.sym('alpha')
    # Define the differential states as CasADi symbols
    # T(t-1) + u_rad(t-1) + u_ahu(t-1) + v_IG(t-1) + u_blinds(t-1) + v(t-1)
    x = MX.sym('x', 16)
    # Define the control inputs as CasADi symbols
    u_blinds_E = MX.sym('u_blinds_E')
    u_blinds_N = MX.sym('u_blinds_N')
    u_blinds_S = MX.sym('u_blinds_S')
    u_blinds_W = MX.sym('u_blinds_W')
    u_AHU = MX.sym('u_AHU')
    u_rad = MX.sym('u_rad')

    u = vertcat(u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u_AHU, u_rad)
    # Define time-varying parameters that can chance at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions
    v_IG = MX.sym('v_IG')
    v_Tamb = MX.sym('v_Tamb')
    v_solGlobFac_S = MX.sym('v_solGlobFac_S')
    v_solGlobFac_W = MX.sym('v_solGlobFac_W')
    v_solGlobFac_N = MX.sym('v_solGlobFac_N')
    v_solGlobFac_E = MX.sym('v_solGlobFac_E')
    v_windspeed = MX.sym('v_windspeed')
    v_Hum_amb = MX.sym('v_Hum_amb')
    v_P_amb = MX.sym('v_P_amb')

    u_ahu_ub = MX.sym('u_ahu_ub')
    setp_ub = MX.sym('setp_ub')
    setp_lb = MX.sym('setp_lb')

    u_blinds_E_val = MX.sym('u_blinds_E_val')
    u_blinds_N_val = MX.sym('u_blinds_N_val')
    u_blinds_S_val = MX.sym('u_blinds_S_val')
    u_blinds_W_val = MX.sym('u_blinds_W_val')

    v = vertcat(v_IG, v_Tamb, v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W, v_windspeed, v_Hum_amb, v_P_amb)

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    if zone == 'Coworking':
        input = vertcat(x, u_blinds_E, u_blinds_N_val, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Corridor':
        input = vertcat(x, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Entrance':
        input = vertcat(x, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'LabNorth':
        input = vertcat(x, u_blinds_E, u_blinds_N_val, u_blinds_S, u_blinds_W_val, u[4:], v)
    elif zone == 'LabSouth':
        input = vertcat(x, u_blinds_E, u_blinds_N, u_blinds_S_val, u_blinds_W_val, u[4:], v)
    elif zone == 'MeetingSouth':
        input = vertcat(x, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'MeetingNorth':
        input = vertcat(x, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Nerdroom1':
        input = vertcat(x, u_blinds_E, u_blinds_N, u_blinds_S_val, u_blinds_W, u[4:], v)
    elif zone == 'Nerdroom2':
        input = vertcat(x, u_blinds_E, u_blinds_N, u_blinds_S_val, u_blinds_W, u[4:], v)
    elif zone == 'RestroomM':
        input = vertcat(x, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'RestroomW':
        input = vertcat(x, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Space01':
        input = vertcat(x, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Stairway':
        input = vertcat(x, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)

    # input = vertcat(x,u,v)
    input = NP.divide(input - x_lb_NN, x_ub_NN - x_lb_NN)

    for i in range(1,len(model.layers) + 1):
        tmp_a = 'a' +  str(i) + '_'
        if i == 1:
            vars()[tmp_a] = input
        vars()[tmp_a] = vertcat(1, vars()[tmp_a])
        tmp_z = 'z' + str(i) + '_'
        vars()[tmp_z] = mtimes(Theta['Theta'+str(i)], vars()[tmp_a])
        tmp_a = 'a' + str(i+1) + '_'
        if i < len(model.layers):
            vars()[tmp_a] = tanh(vars()[tmp_z])
        else:
            vars()[tmp_a] = vars()[tmp_z]

    dT = MX.sym('dT')
    dHeatrate = MX.sym('dHeatrate')
    dT = NP.multiply(vars()[tmp_a][0], y_ub_NN[0]-y_lb_NN[0]) + y_lb_NN[0]
    dHeatrate = NP.multiply(vars()[tmp_a][1], y_ub_NN[1]-y_lb_NN[1]) + y_lb_NN[1]
    #
    if zone == 'Coworking':
        dx = vertcat(dT, u_blinds_E, u_blinds_N_val, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Corridor':
        dx = vertcat(dT, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Entrance':
        dx = vertcat(dT, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'LabNorth':
        dx = vertcat(dT, u_blinds_E, u_blinds_N_val, u_blinds_S, u_blinds_W_val, u[4:], v)
    elif zone == 'LabSouth':
        dx = vertcat(dT, u_blinds_E, u_blinds_N, u_blinds_S_val, u_blinds_W_val, u[4:], v)
    elif zone == 'MeetingSouth':
        dx = vertcat(dT, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'MeetingNorth':
        dx = vertcat(dT, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Nerdroom1':
        dx = vertcat(dT, u_blinds_E, u_blinds_N, u_blinds_S_val, u_blinds_W, u[4:], v)
    elif zone == 'Nerdroom2':
        dx = vertcat(dT, u_blinds_E, u_blinds_N, u_blinds_S_val, u_blinds_W, u[4:], v)
    elif zone == 'RestroomM':
        dx = vertcat(dT, u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'RestroomW':
        dx = vertcat(dT, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Space01':
        dx = vertcat(dT, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)
    elif zone == 'Stairway':
        dx = vertcat(dT, u_blinds_E_val, u_blinds_N, u_blinds_S, u_blinds_W, u[4:], v)


    # dx = vertcat(dT,u,v)
    # Concatenate differential states, algebraic states, control inputs and right-hand-sides
    _x = x

    _z = vertcat([])

    _u = u

    _xdot = dx

    _zdot = vertcat([])

    _p = vertcat(alpha)

    _tv_p = vertcat(v, u_ahu_ub ,setp_ub, setp_lb, u_blinds_E_val, u_blinds_N_val, u_blinds_S_val, u_blinds_W_val)
    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    disturbances = NP.load('Neural_Network\disturbances.npy').squeeze()
    x0 = NP.array([18,0,0,0,0,0,18])
    # the first len(zones)-elements are all v_IG. After those weather information follows
    x0 = NP.concatenate((x0, NP.array([disturbances[zonenumber[zone],0]]), disturbances[len(zones_Heating):,0]))
    # No algebraic states
    z0 = NP.array([])
    # Bounds on the states. Use "inf" for unconstrained states

    disturbances_lb = NP.min(disturbances,axis =1)
    disturbances_ub = NP.max(disturbances,axis =1)

    x_lb = NP.concatenate((NP.array([x_lb_NN[0]]), NP.array([0,0,0,0,0,17]) - 1e-1, NP.array([disturbances_lb[zonenumber[zone]]]) ,disturbances_lb[len(zones_Heating):])) #- 1e-1
    x_ub = NP.concatenate((NP.array([x_ub_NN[0]]), NP.array([1,1,1,1,1,22]) + 1e-1, NP.array([disturbances_ub[zonenumber[zone]]]) ,disturbances_ub[len(zones_Heating):])) #+ 1e-1
    # No algebraic states
    z_lb = NP.array([])
    z_ub = NP.array([])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    u_lb = NP.array([0,0,0,0,0,17])
    u_ub = NP.array([0,0,0,0,1,22])

    if zone == 'Coworking':
        u_ub[0] = 1
    elif zone == 'MeetingNorth':
        u_ub[1] = 1
    elif zone == 'MeetingSouth':
        u_ub[2] = 1
    elif zone == 'Entrance':
        u_ub[3] = 1
    # RestroomM has no window
    elif zone == 'RestroomM':
        u_ub[4] = 0



    u0 = NP.array([0,0,0,0,0,18])
    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.ones(x.shape[0])
    z_scaling = NP.array([])
    u_scaling = NP.array([1,1,1,1,1,1])
    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    # cons = vertcat(x[0:len(zones)]-setp_ub, -(x[0:len(zones)]-setp_lb))
    cons = vertcat(x[0]-setp_ub, -(x[0]-setp_lb))
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    # cons_ub = NP.zeros(2*len(zones))
    cons_ub = NP.zeros(2)

    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 1
    # Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    if zone == 'LabSouth':
        penalty_term_cons = 5e5*NP.ones(2)
    elif zone == 'Entrance':
        penalty_term_cons = 5e5*NP.ones(2)
    elif zone == 'RestroomM':
        penalty_term_cons = 5e5*NP.ones(2)
    elif zone == 'MeetingSouth':
        penalty_term_cons = 5e5*NP.ones(2)
    else:
        penalty_term_cons = 2e5*NP.ones(2)
    # Maximum violation for the constraints
    maximum_violation = NP.array([20,0.5])

    # Define the terminal constraint (leave it empty if not necessary)
    cons_terminal = vertcat()
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_terminal_lb = NP.array([])
    cons_terminal_ub = NP.array([])


    """
    --------------------------------------------------------------------------
    template_model: cost function
    --------------------------------------------------------------------------
    """
    # Define the cost function
    # Penalty term for the control movements 1e4, 100
    if zone == 'MeetingNorth':
        lterm = 0.5*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1e4*NP.ones(1), 50*NP.ones(1)))
    elif zone == 'Coworking':
        lterm = 0.5*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1e4*NP.ones(1), 50*NP.ones(1)))
    elif zone == 'MeetingSouth':
        lterm = 0.01*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 5*1e4*NP.ones(1), 50*NP.ones(1)))
    elif zone == 'Entrance':
        lterm = 3*u_rad + 0.01*dHeatrate
        rterm = NP.concatenate((1e4*NP.ones(4), 5*1e4*NP.ones(1), 55*NP.ones(1)))
    elif zone == 'Corridor':
        lterm = 0.01*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 5*1e4*NP.ones(1), 20*NP.ones(1)))
    elif zone == 'LabSouth':
        lterm = 0.01*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 5*1e4*NP.ones(1), 50*NP.ones(1)))
    elif zone == 'Nerdroom1':
        lterm = 0.5*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1e4*NP.ones(1), 50*NP.ones(1)))
    elif zone == 'Nerdroom2':
        lterm = 0.5*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1e4*NP.ones(1), 50*NP.ones(1)))

    elif zone == 'RestroomM':
        lterm = 4*u_rad + dHeatrate
        rterm = NP.concatenate((0*1e4*NP.ones(4), 0*1e4*NP.ones(1), 55*NP.ones(1)))

    elif zone == 'RestroomW':
        lterm = 0.01*dHeatrate + 1*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1*1e4*NP.ones(1), 50*NP.ones(1)))

    elif zone == 'Stairway':
        lterm = 0.5*dHeatrate + 5*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1*1e4*NP.ones(1), 25*NP.ones(1)))

    elif zone == 'Space01':
        lterm = 0.5*dHeatrate + 2*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1*1e4*NP.ones(1), 10*NP.ones(1)))
    else:
        lterm = 0.5*dHeatrate + 10*u_rad
        rterm = NP.concatenate((1e4*NP.ones(4), 1e4*NP.ones(1), 20*NP.ones(1)))
    mterm = lterm


    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z, 'aes': _zdot,'x0': x0, 'z0':z0, 'x_lb': x_lb,'x_ub': x_ub, 'z_lb': z_lb,'z_ub': z_ub, 'u0':u0,
    'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'z_scaling':z_scaling, 'u_scaling':u_scaling, 'cons':cons, 'tv_p':_tv_p,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub,
    'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation,
    'mterm': mterm,'lterm':lterm, 'rterm':rterm, 'zone_name': zone}
    model = core_do_mpc.model(model_dict)

    return model
