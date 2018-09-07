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

# This is the main path of your do-mpc installation relative to the execution folder
path_do_mpc = '../'
# Add do-mpc path to the current directory
import sys
sys.path.insert(0,path_do_mpc+'code')
# Do not write bytecode to maintain clean directories
sys.dont_write_bytecode = True

# Start CasADi
from casadi import *
# Import do-mpc core functionalities
import core_do_mpc
# Import do-mpc plotting and data managament functions
import data_do_mpc

from pyfmi import load_fmu
from pdb import set_trace as bp
import time
import numpy as NP
from vars import *


"""
-----------------------------------------------
do-mpc: Definition of the do-mpc configuration
-----------------------------------------------
"""

# Import the user defined modules
import template_model
import template_optimizer
import template_observer
import template_simulator
from step_simulator import step_simulator
from step_simulator import compare
from vars import *

configurations = []
for zone in zones:
    # Create the objects for each module
    vars()['model_' + zone] = template_model.model(zone)
    # Create an optimizer object based on the template and a model
    vars()['optimizer_' + zone] = template_optimizer.optimizer(vars()['model_' + zone], zonenumber[zone])
    # Create an observer object based on the template and a model
    vars()['observer_' + zone] = template_observer.observer(vars()['model_' + zone])
    # Create a simulator object based on the template and a model
    vars()['simulator_' + zone] = template_simulator.simulator(vars()['model_' + zone], zonenumber[zone])
    # Create a configuration
    vars()['configuration_' + zone] = core_do_mpc.configuration(vars()['model_' + zone], vars()['optimizer_' + zone], vars()['observer_' + zone], vars()['simulator_' + zone])
    # Set up the solvers
    vars()['configuration_' + zone].setup_solver()

    configurations.append(vars()['configuration_' + zone])
    # remove zone -> only unused zones remain these will be set to 20 degree in step_simulator
    unused_zones.remove(zone)

# Load FMU created from compile_fmu() or EnergyPlusToFMU
modelName = 'RKF_Berlin'
model_fmu = load_fmu(modelName+'.fmu')

# Load options
opts = model_fmu.simulate_options()
# Set number of timesteps
opts['ncp'] = numSteps
# Manually initialize
opts['initialize'] = False
model_fmu.initialize(configurations[0].simulator.t0_sim*60 ,configurations[0].optimizer.t_end*60)
"""
----------------------------
do-mpc: MPC loop
----------------------------
"""

start_time = time.time()
while (configurations[0].simulator.t0_sim + configurations[0].simulator.t_step_simulator < configurations[0].optimizer.t_end):

# int(configurations.simulator.t0_sim / configurations.simulator.t_step_simulator)
    """
    ----------------------------
    do-mpc: Optimizer
    ----------------------------
    """
    # Make one optimizer step (solve the NLP)
    for zone in zones:
        vars()['configuration_' + zone].make_step_optimizer()

        step_index = int(vars()['configuration_' + zone].simulator.t0_sim / vars()['configuration_' + zone].simulator.t_step_simulator)
        v_opt = vars()['configuration_' + zone].optimizer.opt_result_step.optimal_solution
        U_offset = vars()['configuration_' + zone].optimizer.nlp_dict_out['U_offset']
        # Set the optimal blind position as tv_p for other zones - if simulating the whole model you can replace zones with eastSide, northSide, etc.
        if zone == 'Coworking':
            for k in eastSide: vars()['configuration_' + k].optimizer.tv_p_values[step_index,-4,:] = NP.round(NP.squeeze(v_opt[U_offset]))
        elif zone == 'MeetingNorth':
            for k in northSide: vars()['configuration_' + k].optimizer.tv_p_values[step_index,-3,:] = NP.round(NP.squeeze(v_opt[U_offset + 1]))
        elif zone == 'MeetingSouth':
            for k in southSide: vars()['configuration_' + k].optimizer.tv_p_values[step_index,-2,:] = NP.round(NP.squeeze(v_opt[U_offset + 2]))
        elif zone == 'Entrance':
            for k in westSide: vars()['configuration_' + k].optimizer.tv_p_values[step_index,-1,:] = NP.round(NP.squeeze(v_opt[U_offset + 3]))
        # print vars()['configuration_' + zone].optimizer.tv_p_values[step_index,-7:-4,:]
    """
    ----------------------------
    do-mpc: Simulator
    ----------------------------
    """
    # Simulate the system one step using the solution obtained in the optimization
    current_t = configurations[0].simulator.t0_sim
    tstep = configurations[0].simulator.t_step_simulator
    step_simulator(model_fmu, current_t, tstep, configurations)

    # compare(model_fmu, current_t, tstep, configurations)


    for zone in zones:
        """
        ----------------------------
        do-mpc: Observer
        ----------------------------
        """
        # Make one observer step
        vars()['configuration_' + zone].make_step_observer()

        """
        ------------------------------------------------------
        do-mpc: Prepare next iteration and store information
        ------------------------------------------------------
        """
        # Store the information
        vars()['configuration_' + zone].store_mpc_data()

        # Set initial condition constraint for the next iteration
        vars()['configuration_' + zone].prepare_next_iter()

        """
        ------------------------------------------------------
        do-mpc: Plot MPC animation if chosen by the user
        ------------------------------------------------------
        """
        # Plot animation if chosen in by the user
        data_do_mpc.plot_animation(vars()['configuration_' + zone])
"""
------------------------------------------------------
do-mpc: Plot the closed-loop results
------------------------------------------------------
"""
elapsed_time = time.time() - start_time
#print elapsed_time
print 'Elapsed Time: ' + str(elapsed_time)
for i in range(len(configurations)):
    consumedpower = NP.sum(configurations[i].simulator.Heatrate)
    print "Consumed Power: " + str(consumedpower) + " [W]"

    data_do_mpc.plot_mpc(configurations[i])

    # Export to matlab if wanted
    data_do_mpc.export_to_matlab(configurations[i])
    data_do_mpc.save_simulation(configurations[i])

raw_input("Press Enter to exit do-mpc...")
