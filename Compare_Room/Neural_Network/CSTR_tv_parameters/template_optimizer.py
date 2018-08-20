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
from pdb import set_trace as bp
from vars import *

def createConstraints(ub_night, lb_night, ub_day, lb_day, dist_len):
    daycount = 0
    constraints = NP.zeros((3,dist_len))
    for index in range (dist_len):
        if index != 0 and index % 144 == 0:
            daycount += 1
        if index - daycount*144 < 7*EPTimeStep or index - daycount*144 > 20*EPTimeStep:
            constraints[0, index] = 0
            constraints[1, index] = ub_night
            constraints[2, index] = lb_night
            ## FIXME: do variable constraint for window -> no opening during the night
        elif index - daycount*144 < 7*(EPTimeStep) + 8:
            constraints[0, index] = 0
            constraints[1, index] = ub_day
            constraints[2, index] = lb_night
        elif index - daycount*144 > 20*EPTimeStep - 8:
            constraints[0, index] = 0
            constraints[1, index] = ub_day
            constraints[2, index] = lb_night
        else:
            constraints[0, index] = 1
            constraints[1, index] = ub_day
            constraints[2, index] = lb_day
    return constraints

def optimizer(model):

    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """

    # Prediction horizon
    #NOTE in setup_nlp Changed from TV_P to TV_P[:,k]
    n_horizon = 10
    # Robust horizon, set to 0 for standard NMPC
    n_robust = 0
    # open_loop robust NMPC (1) or multi-stage NMPC (0). Only important if n_robust > 0
    open_loop = 0
    t_end = days*hours*minutes + daystart       # Number of minutes
    t_step = minutes/EPTimeStep                 # 10 min step

    # Choose type of state discretization (collocation or multiple-shooting)
    state_discretization = 'discrete-time'
    # Degree of interpolating polynomials: 1 to 5
    poly_degree = 2
    # Collocation points: 'legendre' or 'radau'
    collocation = 'radau'
    # Number of finite elements per control interval
    n_fin_elem = 2
    # NLP Solver and linear solver
    nlp_solver = 'ipopt'
    qp_solver = 'qpoases'

    # It is highly recommended that you use a more efficient linear solver
    # such as the hsl linear solver MA27, which can be downloaded as a precompiled
    # library and can be used by IPOPT on run time

    linear_solver = 'mumps' # 'MA27'

    # GENERATE C CODE shared libraries NOTE: Not currently supported
    generate_code = 0

    """
    --------------------------------------------------------------------------
    template_optimizer: uncertain parameters
    --------------------------------------------------------------------------
    """
    # Define the different possible values of the uncertain parameters in the scenario tree
    alpha_values = NP.array([1,1])
    # beta_values = NP.array([0,0])
    uncertainty_values = NP.array([alpha_values])
    # Parameteres of the NLP which may vary along the time (For example a set point that varies at a given time)
    set_point = SX.sym('set_point')
    parameters_nlp = NP.array([set_point])

    """
    --------------------------------------------------------------------------
    template_optimizer: time-varying parameters
    --------------------------------------------------------------------------
    """
    # Only necessary if time-varying paramters defined in the model
    # The length of the vector for each parameter should be the prediction horizon
    # The vectos for each parameter might chance at each sampling time
    number_steps = numSteps/days*364
    # Number of time-varying parameters
    n_tv_p = 12
    tv_p_values = NP.resize(NP.array([]),(number_steps,n_tv_p,n_horizon))
    disturbances = NP.load('Neural_Network\disturbances.npy').squeeze()

    # Create a constraint vector - night constraints - day constraints
    constraints = createConstraints(19,17,22,20, disturbances.shape[1])
    # constraints = createConstraints(23,21,23,21, disturbances.shape[1])
    for time_step in range (number_steps):
        tv_param_1_values = disturbances[0, time_step: time_step+n_horizon]
        tv_param_2_values = disturbances[1, time_step: time_step+n_horizon]
        tv_param_3_values = disturbances[2, time_step: time_step+n_horizon]
        tv_param_4_values = disturbances[3, time_step: time_step+n_horizon]
        tv_param_5_values = disturbances[4, time_step: time_step+n_horizon]
        tv_param_6_values = disturbances[5, time_step: time_step+n_horizon]
        tv_param_7_values = disturbances[6, time_step: time_step+n_horizon]
        tv_param_8_values = disturbances[7, time_step: time_step+n_horizon]
        tv_param_9_values = disturbances[8, time_step: time_step+n_horizon]

        tv_param_10_values = constraints[0, time_step: time_step+n_horizon]
        tv_param_11_values = constraints[1, time_step: time_step+n_horizon]
        tv_param_12_values = constraints[2, time_step: time_step+n_horizon]

        tv_p_values[time_step] = NP.array([tv_param_1_values,tv_param_2_values,tv_param_3_values,
        tv_param_4_values,tv_param_5_values,tv_param_6_values, tv_param_7_values,
        tv_param_8_values, tv_param_9_values,tv_param_10_values,
        tv_param_11_values, tv_param_12_values])
    # Parameteres of the NLP which may vary along the time (For example a set point that varies at a given time)
    """
    --------------------------------------------------------------------------
    template_optimizer: pass_information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    # Check if the user has introduced the data correctly
    optimizer_dict = {'n_horizon':n_horizon, 'n_robust':n_robust, 't_step': t_step,
    't_end':t_end,'poly_degree': poly_degree, 'collocation':collocation,
    'n_fin_elem': n_fin_elem,'generate_code':generate_code,'open_loop': open_loop,
    'uncertainty_values':uncertainty_values,'parameters_nlp':parameters_nlp,
    'state_discretization':state_discretization,'nlp_solver': nlp_solver,
    'linear_solver':linear_solver, 'qp_solver':qp_solver, 'tv_p_values':tv_p_values}
    optimizer_1 = core_do_mpc.optimizer(model,optimizer_dict)
    return optimizer_1
