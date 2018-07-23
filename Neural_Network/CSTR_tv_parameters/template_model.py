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

def relu(x):
    return tanh(x)#fmax(0,x)

def model():
    lbub = NP.load('Neural_Network\lbub.npy')
    x_lb_NN = lbub[0]
    x_ub_NN = lbub[1]
    y_lb_NN = lbub[2]
    y_ub_NN = lbub[3]


    json_file = open('Neural_Network\model_3.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("Neural_Network\model_3.h5")
    Theta = {}
    # for i in range(len(model.layers)):
    i = 0
    for j in [0,2,4,6,8]:
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

    # x = MX.sym("x", Theta['Theta1'].shape[1]-1)
    features = 15
    numbers = 2 - 1
    x = MX.sym("x", features*numbers)

    # Define the algebraic states as CasADi symbols

    # Define the control inputs as CasADi symbols

    u_rad_OfficesZ1 = MX.sym("u_rad_OfficesZ1")
    u_AHU1_noERC = MX.sym("u_AHU1_noERC")
    u_blinds_E = MX.sym("u_blinds_E")
    u_blinds_N = MX.sym("u_blinds_N")
    u_blinds_S = MX.sym("u_blinds_S")
    u_blinds_W = MX.sym("u_blinds_W")

    u = vertcat(u_blinds_E, u_blinds_N, u_blinds_S, u_blinds_W, u_AHU1_noERC, u_rad_OfficesZ1)

    # Define time-varying parameters that can chance at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions
    v_IG_Offices = MX.sym("V_IG_Offices")
    v_Tamb = MX.sym("v_Tamb")
    v_solGlobFac_S = MX.sym("v_solGlobFac_S")
    v_solGlobFac_W = MX.sym("v_solGlobFac_W")
    v_solGlobFac_N = MX.sym("v_solGlobFac_N")
    v_solGlobFac_E = MX.sym("v_solGlobFac_E")
    v_windspeed = MX.sym("v_windspeed")
    setp_ub = MX.sym("setp_ub")
    setp_lb = MX.sym("setp_lb")

    u_ahu_ub = MX.sym("u_ahu_ub")

    v = vertcat(v_IG_Offices, v_Tamb, v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W, v_windspeed, u_ahu_ub ,setp_ub, setp_lb)


    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    input = vertcat(x,u,v[0:7])
    input = NP.divide(input - x_lb_NN, x_ub_NN - x_lb_NN)#(input - x_lb_NN)*float(1)/(x_ub_NN - x_lb_NN)
    a1 = input
    a1 = vertcat(1,a1)
    z1 = mtimes(Theta['Theta1'],a1)
    a2 = tanh(z1)
    a2 = vertcat(1,a2)

    z2 = mtimes(Theta['Theta2'],a2)
    a3 = tanh(z2)
    a3 = vertcat(1,a3)

    z3 = mtimes(Theta['Theta3'],a3)
    a4 = tanh(z3)
    a4 = vertcat(1,a4)

    z4 = mtimes(Theta['Theta4'],a4)
    a5 = tanh(z4)
    a5 = vertcat(1,a5)

    z5 = mtimes(Theta['Theta5'],a5)
    a6 = z5

    dx1 = MX.sym("dx1")
    dx2 = MX.sym("dx2")

    dx1 = NP.multiply(a6[0], y_ub_NN[0]-y_lb_NN[0]) + y_lb_NN[0]#dx*(y_ub_NN-y_lb_NN) + y_lb_NN
    dx2 = NP.multiply(a6[1], y_ub_NN[1]-y_lb_NN[1]) + y_lb_NN[1]
    dx = vertcat(dx1, dx2,u,v[0:7])
    # Concatenate differential states, algebraic states, control inputs and right-hand-sides
    _x = x

    _z = vertcat([])

    _u = u

    _xdot = dx

    _zdot = vertcat([])

    # _p = vertcat(alpha, beta)
    _p = vertcat(alpha)

    # _tv_p = vertcat([])
    _tv_p = vertcat(v)

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    x0 = NP.array([18,0,0,0,0,0,0,18,141,5,0,0,0,0,2])
    # No algebraic states
    z0 = NP.array([])
    # Bounds on the states. Use "inf" for unconstrained states
    x_lb = -20000* NP.ones(features*numbers)#x_lb_NN[0:features*numbers]
    x_lb[1] = 0
    x_ub = 20000* NP.ones(features*numbers)#x_ub_NN[0:features*numbers]
    # No algebraic states
    z_lb = NP.array([])
    z_ub = NP.array([])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    u_lb = NP.array([0,0,0,0, 0, 16])
    u_ub = NP.array([1,1,1,1, 1, 23])
    u0 = NP.array([0,0,0,0, 0, 21])
    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.ones(features*numbers)
    z_scaling = NP.array([])
    u_scaling = NP.array([1,1,1,1,1,1])
    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat(x[features*(numbers-1)]-v[-2], -(x[features*(numbers-1)]-v[-1]))
    # cons = vertcat(x[0], -x[0])
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_ub = NP.array([0, -0])

    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 1
    # Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([1e5, 1e5])
    # Maximum violation for the constraints
    maximum_violation = 100*NP.array([1, 1])

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
    # Lagrange term
    # Mayer term
    lterm = 1*u_rad_OfficesZ1 + 0.1*x[1] #+ u_AHU1_noERC   # + (u_rad_OfficesZ1-18)*u_AHU1_noERC  #+ u_blinds_E + u_blinds_N + u_blinds_S + u_blinds_W # - (u_AHU1_noERC-25)**2  #(5-u_rad_OfficesZ1)**2 + (5-u_rad_OfficesZ2)**2 + (0.7-u_blinds_S)**2 + (0.7-u_blinds_W)**2 + (0.7-u_blinds_N)**2 + (0.7-u_blinds_E)**2 + (0.1-u_AHU1_noERC)**2 + (0.1-u_AHU2_noERC)**2
    mterm = 1*u_rad_OfficesZ1 + 0.1*x[1] #+ u_AHU1_noERC  # + (u_rad_OfficesZ1-18)*u_AHU1_noERC  #+ u_blinds_E + u_blinds_N + u_blinds_S + u_blinds_W #- (u_AHU1_noERC-25)**2 #(5-u_rad_OfficesZ1)**2 + (5-u_rad_OfficesZ2)**2 + (0.7-u_blinds_S)**2 + (0.7-u_blinds_W)**2 + (0.7-u_blinds_N)**2 + (0.7-u_blinds_E)**2 + (0.1-u_AHU1_noERC)**2 + (0.1-u_AHU2_noERC)**2
    # Penalty term for the control movements 1e4, 100
    rterm =5e1* NP.array([1, 1, 1, 1, 1e-1, 1e-1])
    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z, 'aes': _zdot,'x0': x0, 'z0':z0, 'x_lb': x_lb,'x_ub': x_ub, 'z_lb': z_lb,'z_ub': z_ub, 'u0':u0,
    'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'z_scaling':z_scaling, 'u_scaling':u_scaling, 'cons':cons, 'tv_p':_tv_p,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}
    model = core_do_mpc.model(model_dict)

    return model
