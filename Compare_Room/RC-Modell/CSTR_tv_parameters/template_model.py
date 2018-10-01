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
from pdb import set_trace as bp


def model():
    # Load Discrete_Time_Model
    Discrete_Time_Model = scipy.io.loadmat('RKF_Discrete_Time_Model.mat')

    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    alpha   = SX.sym("alpha")
    # beta    = SX.sym("beta")
    # Define the differential states as CasADi symbols

    x = SX.sym("x", 37)

    # Define the algebraic states as CasADi symbols

    # Define the control inputs as CasADi symbols

    u_rad_OfficesZ1 = SX.sym("u_rad_OfficesZ1")

    u_AHU1_noERC = SX.sym("u_AHU1_noERC")

    u_blinds_N = SX.sym("u_blinds_N")
    u_blinds_W = SX.sym("u_blinds_W")


    u = vertcat(u_blinds_N, u_blinds_W, u_AHU1_noERC, u_rad_OfficesZ1)

    # Define time-varying parameters that can chance at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions
    v_IG_Offices = SX.sym("V_IG_Offices")
    v_Tamb = SX.sym("v_Tamb")
    v_Tgnd = SX.sym("v_Tgnd")
    v_solGlobFac_S = SX.sym("v_solGlobFac_S")
    v_solGlobFac_W = SX.sym("v_solGlobFac_W")
    v_solGlobFac_N = SX.sym("v_solGlobFac_N")
    v_solGlobFac_E = SX.sym("v_solGlobFac_E")
    v_windspeed = SX.sym('v_windspeed')
    setp_ub = SX.sym("setp_ub")
    setp_lb = SX.sym("setp_lb")
    u_ahu_ub = SX.sym("u_ahu_ub")

    v = vertcat(v_IG_Offices, v_Tamb, v_Tgnd, v_solGlobFac_E, v_solGlobFac_N, v_solGlobFac_S, v_solGlobFac_W, v_windspeed, u_ahu_ub, setp_ub, setp_lb)


    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """

    # Define the differential equations
    Bvu = Discrete_Time_Model["Bvu"]
    Bxu = Discrete_Time_Model["Bxu"]


    dx = mtimes(Discrete_Time_Model["A"], x) + mtimes(Discrete_Time_Model["Bu"], u) + mtimes(Discrete_Time_Model["Bv"], v[0:7])
    sum = 0
    for i in range(u.shape[0]):
        sum = sum + (mtimes(Bvu[:,:,i], v[0:7]) + mtimes(Bxu[:,:,i], x)) * u[i]
    dx = dx + sum
    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = x

    _z = vertcat([])

    _u = u

    _xdot = dx

    _zdot = vertcat([])

    # _p = vertcat(alpha, beta)
    _p = vertcat(alpha)

    # _tv_p = vertcat([])
    _tv_p = v

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    x0 = 18*NP.ones(37)
    x0[[3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35]] = 5
    # No algebraic states
    z0 = NP.array([])

    # Bounds on the states. Use "inf" for unconstrained states
    x_lb = -50*NP.ones(37)
    x_ub = 60*NP.ones(37)
    # No algebraic states
    z_lb = NP.array([])
    z_ub = NP.array([])


    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    u_lb = NP.array([0, 0, 0, 0])
    u_ub = NP.array([1, 1, 0.2016, inf])
    u0 = NP.array([1, 1, 0, 0])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.ones(37)
    z_scaling = NP.array([])
    u_scaling = NP.array([1,1,1,1])
    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat(x[0]-v[-2], -(x[0]-v[-1]))
    # cons = vertcat(x[0], x[1], -x[0], -x[1])
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
    lterm = u_rad_OfficesZ1  + u_AHU1_noERC  #+ u_blinds_E + u_blinds_N + u_blinds_S + u_blinds_W #(5-u_rad_OfficesZ1)**2 + (5-u_rad_OfficesZ2)**2 + (0.7-u_blinds_S)**2 + (0.7-u_blinds_W)**2 + (0.7-u_blinds_N)**2 + (0.7-u_blinds_E)**2 + (0.1-u_AHU1_noERC)**2 + (0.1-u_AHU2_noERC)**2
    mterm = u_rad_OfficesZ1  + u_AHU1_noERC  #+ u_blinds_E + u_blinds_N + u_blinds_S + u_blinds_W #(5-u_rad_OfficesZ1)**2 + (5-u_rad_OfficesZ2)**2 + (0.7-u_blinds_S)**2 + (0.7-u_blinds_W)**2 + (0.7-u_blinds_N)**2 + (0.7-u_blinds_E)**2 + (0.1-u_AHU1_noERC)**2 + (0.1-u_AHU2_noERC)**2
    # Penalty term for the control movements
    rterm =1e-3 * NP.array([1, 1, 1, 1])
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
