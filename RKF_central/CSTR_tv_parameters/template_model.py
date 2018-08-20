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

def model():
    lbub = NP.load('Neural_Network\lbub_60.npy')
    x_lb_NN = lbub[0]
    x_ub_NN = lbub[1]
    y_lb_NN = lbub[2]
    y_ub_NN = lbub[3]


    json_file = open('Neural_Network\model_60.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("Neural_Network\model_60.h5")
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

    alpha   = MX.sym("alpha")
    # Define the differential states as CasADi symbols
    # T(t-1) + u_rad(t-1) + u_ahu(t-1) + v_IG(t-1) + u_blinds(t-1) + v(t-1)
    #+ len(zones_Heating) + len(zones_ahu) + len(zones)
    x = MX.sym("x", (numbers-1)*(len(zones)  + 12))
    # Define the control inputs as CasADi symbols
    u_blinds_E = MX.sym("u_blinds_E")
    u_blinds_N = MX.sym("u_blinds_N")
    u_blinds_S = MX.sym("u_blinds_S")
    u_blinds_W = MX.sym("u_blinds_W")

    u = vertcat(u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W)

    for i in range(len(zones_ahu)):
        tmp = 'u_AHU_' + zones_ahu[i]
        vars()[tmp] = MX.sym(tmp)
        u = vertcat(u,vars()[tmp])


    for i in range(len(zones_Heating)):
        tmp = 'u_rad_' + zones_Heating[i]
        vars()[tmp] = MX.sym(tmp)
        u = vertcat(u,vars()[tmp])
        if i == 0:
            u_rad = MX.sym('u_rad')
            u_rad = vars()[tmp]
        else:
            u_rad = vertcat(u_rad,vars()[tmp])

    # Define time-varying parameters that can chance at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions
    for i in range(len(zones)):
        tmp = 'v_IG_' + zones[i]
        vars()[tmp] = MX.sym(tmp)
        if i == 0:
            v = vars()[tmp]
        else:
            v = vertcat(v,vars()[tmp])


    v_Tamb = MX.sym("v_Tamb")
    v_solGlobFac_S = MX.sym("v_solGlobFac_S")
    v_solGlobFac_W = MX.sym("v_solGlobFac_W")
    v_solGlobFac_N = MX.sym("v_solGlobFac_N")
    v_solGlobFac_E = MX.sym("v_solGlobFac_E")
    v_windspeed = MX.sym("v_windspeed")
    v_Hum_amb = MX.sym("v_Hum_amb")
    v_P_amb = MX.sym("v_P_amb")

    u_ahu_ub = MX.sym("u_ahu_ub")
    setp_ub = MX.sym("setp_ub")
    setp_lb = MX.sym("setp_lb")

    v = vertcat(v, v_Tamb, v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W, v_windspeed, v_Hum_amb, v_P_amb)

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    input = vertcat(x,u,v)
    input = NP.divide(input - x_lb_NN, x_ub_NN - x_lb_NN)

    for i in range(1,len(model.layers)):
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
    for i in range(len(zones)  + len(zones_Heating)):
        if i < len(zones):
            tmp = 'dT_' + zones[i]
        else:
            tmp = 'dHeatrate_' + zones_Heating[i-len(zones)]

        vars()[tmp] = MX.sym(tmp)
        vars()[tmp] = vars()[tmp_a][i]
        vars()[tmp] = NP.multiply(vars()[tmp_a][i], y_ub_NN[i]-y_lb_NN[i]) + y_lb_NN[i]

        if i == 0:
            dT = vars()[tmp]
        elif i < len(zones):
            dT = vertcat(dT,vars()[tmp])
        elif i == len(zones):
            dHeatrate = vars()[tmp]
        else:
            dHeatrate = vertcat(dHeatrate,vars()[tmp])

    # dT = vars()[tmp_a][0:14]
    dx = vertcat(dT,u[0:4],v[len(zones):])
    # Concatenate differential states, algebraic states, control inputs and right-hand-sides
    _x = x

    _z = vertcat([])

    _u = u

    _xdot = dx

    _zdot = vertcat([])

    _p = vertcat(alpha)

    _tv_p = vertcat(v, u_ahu_ub ,setp_ub, setp_lb)
    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    disturbances = NP.load('Neural_Network\disturbances.npy').squeeze()
    x0 = NP.concatenate( (18*NP.ones(len(zones)), NP.zeros(4), disturbances[len(zones):,0] ))
    # No algebraic states
    z0 = NP.array([])
    # Bounds on the states. Use "inf" for unconstrained states

    disturbances_lb = NP.min(disturbances,axis =1)
    disturbances_ub = NP.max(disturbances,axis =1)
    x_lb = NP.concatenate((x_lb_NN[0:len(zones) + 0*len(zones_Heating) + 0*len(zones_ahu)] - 1e-2, NP.zeros(4) ,disturbances_lb[len(zones):]))
    x_ub = NP.concatenate((x_ub_NN[0:len(zones) + 0*len(zones_Heating) + 0*len(zones_ahu)] + 1e-2, NP.ones(4) ,disturbances_ub[len(zones):]))
    # No algebraic states
    z_lb = NP.array([])
    z_ub = NP.array([])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    u_lb = NP.array([0, 0, 0, 0])
    u_lb = NP.concatenate((u_lb, NP.zeros(len(zones_ahu)), 16*NP.ones(len(zones_Heating))))
    u_ub = NP.array([1, 1, 1, 1])
    u_ub = NP.concatenate((u_ub, NP.ones(len(zones_ahu)), 22*NP.ones(len(zones_Heating))))
    u0 = NP.concatenate((NP.zeros(4 + len(zones_ahu)), 18*NP.ones(len(zones_Heating))))

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.ones(x.shape[0])
    z_scaling = NP.array([])
    u_scaling = NP.ones(4 + len(zones_ahu) + len(zones_Heating))
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
    penalty_term_cons = 1e6*NP.ones(2)
    # Maximum violation for the constraints
    maximum_violation = 20*NP.ones(2)

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
    lterm = mtimes(0.5*NP.ones((dHeatrate.shape[1],dHeatrate.shape[0])) , dHeatrate) + mtimes(10*NP.ones((u_rad.shape[1],u_rad.shape[0])) , u_rad)
    mterm = mtimes(0.5*NP.ones((dHeatrate.shape[1],dHeatrate.shape[0])) , dHeatrate) + mtimes(10*NP.ones((u_rad.shape[1],u_rad.shape[0])) , u_rad)
    # Penalty term for the control movements 1e4, 100
    rterm = NP.concatenate((1e4*NP.ones(4), 1e4*NP.ones(len(zones_ahu)), 20*NP.ones(len(zones_Heating))))
    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    bp()
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z, 'aes': _zdot,'x0': x0, 'z0':z0, 'x_lb': x_lb,'x_ub': x_ub, 'z_lb': z_lb,'z_ub': z_ub, 'u0':u0,
    'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'z_scaling':z_scaling, 'u_scaling':u_scaling, 'cons':cons, 'tv_p':_tv_p,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}
    model = core_do_mpc.model(model_dict)

    return model
