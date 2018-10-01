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
from pdb import set_trace as bp
import setup_nlp
from casadi import *
from casadi.tools import *
import data_do_mpc
import numpy as NP
import pdb
import time

import pylab as P

from vars import *

class ocp:
    """ A class that contains a full description of the optimal control problem and will be used in the model class. This is dependent on a specific element of a model class"""
    def __init__(self, param_dict, *opt):
        # Initial state and initial input
        x0 = param_dict["x0"]
        z0 = param_dict["z0"]
        self.x0 = vertcat(x0,z0)
        self.u0 = param_dict["u0"]
        # Bounds for the states
        x_lb = param_dict["x_lb"]
        x_ub = param_dict["x_ub"]
        z_lb = param_dict["z_lb"]
        z_ub = param_dict["z_ub"]
        self.x_lb = vertcat(x_lb,z_lb)
        self.x_ub = vertcat(x_ub,z_ub)
        # Bounds for the inputs
        self.u_lb = param_dict["u_lb"]
        self.u_ub = param_dict["u_ub"]
        # Scaling factors
        x_scaling = param_dict["x_scaling"]
        z_scaling = param_dict["z_scaling"]
        self.x_scaling = vertcat(x_scaling,z_scaling)
        self.u_scaling = param_dict["u_scaling"]
        # Symbolic nonlinear constraints
        self.cons = param_dict["cons"]
        # Upper bounds (no lower bounds for nonlinear constraints)
        self.cons_ub = param_dict["cons_ub"]
        # Terminal constraints
        self.cons_terminal = param_dict["cons_terminal"]
        self.cons_terminal_lb = param_dict["cons_terminal_lb"]
        self.cons_terminal_ub = param_dict["cons_terminal_ub"]
        # Flag for soft constraints
        self.soft_constraint = param_dict["soft_constraint"]
        # Penalty term and maximum violation of soft constraints
        self.penalty_term_cons = param_dict["penalty_term_cons"]
        self.maximum_violation = param_dict["maximum_violation"]
        # Lagrange term, Mayer term, and term for input variations
        self.lterm = param_dict["lterm"]
        self.mterm = param_dict["mterm"]
        self.rterm = param_dict["rterm"]

class model:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, param_dict, *opt):
        # Assert for define length of param_dict
        required_dimension = 30
        if not (len(param_dict) == required_dimension):            raise Exception("Model / OCP information is incomplete. The number of elements in the dictionary is not correct")
        # Assign the main variables describing the model equations
        x = param_dict["x"]
        z = param_dict["z"]
        ode = param_dict["rhs"]
        aes  = param_dict["aes"]
        # Right hand side of the DAE equations
        if z.size(1) == 0:
            self.x = param_dict["x"]
            self.rhs =  param_dict["rhs"]
        else:
            self.x = vertcat(x,z)
            self.rhs = vertcat(ode,aes)
        self.u = param_dict["u"]
        self.p = param_dict["p"]
        self.z = param_dict["z"]
        self.ode = param_dict["rhs"]
        self.aes = param_dict["aes"]
        self.rhs = vertcat(ode,aes)
        self.tv_p = param_dict["tv_p"]
         # Assign the main variables that describe the OCP
        self.ocp = ocp(param_dict)

    @classmethod
    def user_model(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined model class"
        dummy = 1
        return cls(dummy)

class simulator:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, model_simulator, param_dict, *opt):
        # Assert for define length of param_dict
        required_dimension = 11
        if not (len(param_dict) == required_dimension): raise Exception("Simulator information is incomplete. The number of elements in the dictionary is not correct")
        # Unscale the states on the rhs
        rhs_unscaled = substitute(model_simulator.rhs, model_simulator.x, model_simulator.x * model_simulator.ocp.x_scaling)/model_simulator.ocp.x_scaling
        rhs_unscaled = substitute(rhs_unscaled, model_simulator.u, model_simulator.u * model_simulator.ocp.u_scaling)
        # Determine sizes of the state vectors
        nx = model_simulator.x.size(1)
        nz = model_simulator.z.size(1)
        if nz == 0:
            dae = {'x':model_simulator.x, 'p':vertcat(model_simulator.u,model_simulator.p,model_simulator.tv_p), 'ode':model_simulator.ode}
        else:
            dae = {'x':model_simulator.x[0:nx-nz], 'z':model_simulator.z, 'p':vertcat(model_simulator.u,model_simulator.p,model_simulator.tv_p), 'ode':model_simulator.ode, 'alg': model_simulator.aes}

        opts = param_dict["integrator_opts"]
        #NOTE: Check the scaling factors (appear to be fine)
        simulator_do_mpc = integrator("simulator", param_dict["integration_tool"], dae,  opts)
        self.simulator = simulator_do_mpc
        self.plot_states = param_dict["plot_states"]
        self.plot_control = param_dict["plot_control"]
        self.plot_anim = param_dict["plot_anim"]
        self.export_to_matlab = param_dict["export_to_matlab"]
        self.save_simulation = param_dict["save_simulation"]
        self.export_name = param_dict["export_name"]
        self.p_real_now = param_dict["p_real_now"]
        self.tv_p_real_now = param_dict["tv_p_real_now"]
        self.t_step_simulator = param_dict["t_step_simulator"]
        self.t0_sim = daystart#86400*150/60
        self.tf_sim = param_dict["t_step_simulator"] + self.t0_sim
        # NOTE:  The same initial condition than for the optimizer is imposed
        self.x0_sim = model_simulator.ocp.x0 / model_simulator.ocp.x_scaling
        self.xf_sim = 0
        # This is an index to account for the MPC iteration. Starts at 1
        self.mpc_iteration = 1
        # Store HeatRate used in Energpylus
        self.HeatRate = 0*NP.ones((1,1))
        self.faultDetector = 0*NP.ones((1,1))
        self.unmetHours = 0*NP.ones((1,1))
    @classmethod
    def user_simulator(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined simulator class"
        dummy = 1
        return cls(dummy)

    @classmethod
    def application(cls, param_dict, *opt):
        " This is open for the implementation of connection to a real plant"
        dummy = 1
        return cls(dummy)

class optimizer:
    '''This is a class that defines a do-mpc optimizer. The class uses a local model, which
    can be defined independetly from the other modules. The parameters '''
    def __init__(self, optimizer_model, param_dict, *opt):
        # Set the local model to be used by the model
        self.optimizer_model = optimizer_model
        # Assert for the required size of the parameters
        required_dimension = 16
        if not (len(param_dict) == required_dimension): raise Exception("The length of the parameter dictionary is not correct!")
        # Define optimizer parameters
        self.n_horizon = param_dict["n_horizon"]
        self.t_step = param_dict["t_step"]
        self.n_robust = param_dict["n_robust"]
        self.state_discretization = param_dict["state_discretization"]
        self.poly_degree = param_dict["poly_degree"]
        self.collocation = param_dict["collocation"]
        self.n_fin_elem = param_dict["n_fin_elem"]
        self.generate_code = param_dict["generate_code"]
        self.open_loop = param_dict["open_loop"]
        self.t_end = param_dict["t_end"]
        self.nlp_solver = param_dict["nlp_solver"]
        self.linear_solver = param_dict["linear_solver"]
        self.qp_solver = param_dict["qp_solver"]
        # Define model uncertain parameters
        self.uncertainty_values = param_dict["uncertainty_values"]
        # Defin time varying optimizer parameters
        self.tv_p_values = param_dict["tv_p_values"]
        self.parameters_nlp = param_dict["parameters_nlp"]
        # Initialize empty methods for completion later
        self.solver = []
        self.arg = []
        self.nlp_dict_out = []
        self.opt_result_step = []
        self.u_mpc = optimizer_model.ocp.u0
    @classmethod
    def user_optimizer(cls, optimizer_model, param_dict, *opt):
        "This method is open for the impelmentation of a user defined optimizer"
        dummy = 1
        return cls(dummy)

class observer:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, model_observer, param_dict, *opt):
        self.x = param_dict['x']
    @classmethod
    def user_observer(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined estimator class"
        dummy = 1
        return cls(dummy)

class configuration:
    """ A class for the definition of a do-mpc configuration that
    contains a model, optimizer, observer and simulator module """
    def __init__(self, model, optimizer, observer, simulator):
        # The four modules
        self.model = model
        self.optimizer = optimizer
        self.observer = observer
        self.simulator = simulator
        # The data structure
        self.mpc_data = data_do_mpc.mpc_data(self)

    def setup_solver(self):
        # Call setup_nlp to generate the NLP
        nlp_dict_out = setup_nlp.setup_nlp(self.model, self.optimizer)
        # Set options
        opts = {}
        opts["expand"] = True
        opts["ipopt.linear_solver"] = self.optimizer.linear_solver
        #NOTE: this could be passed as parameters of the optimizer class
        opts["ipopt.max_iter"] = 500
        opts["ipopt.tol"] = 1e-6
        opts["ipopt.print_level"] = 0
        # Setup the solver
        opts["print_time"] = False
        start_time = time.time()
        solver = nlpsol("solver", self.optimizer.nlp_solver, nlp_dict_out['nlp_fcn'], opts)
        elapsed_time = time.time() - start_time
        arg = {}
        # Initial condition
        arg["x0"] = nlp_dict_out['vars_init']
        # Bounds on x
        arg["lbx"] = nlp_dict_out['vars_lb']
        arg["ubx"] = nlp_dict_out['vars_ub']
        # Bounds on g
        arg["lbg"] = nlp_dict_out['lbg']
        arg["ubg"] = nlp_dict_out['ubg']
        # NLP parameters
        nu = self.model.u.size(1)
        ntv_p = self.model.tv_p.size(1)
        nk = self.optimizer.n_horizon
        parameters_setup_nlp = struct_symMX([entry("uk_prev",shape=(nu)), entry("TV_P",shape=(ntv_p,nk))])
        param = parameters_setup_nlp(0)
        # First value of the nlp parameters
        param["uk_prev"] = self.model.ocp.u0
        param["TV_P"] = self.optimizer.tv_p_values[0]
        arg["p"] = param
        # Add new attributes to the optimizer class
        self.optimizer.solver = solver
        self.optimizer.arg = arg
        self.optimizer.nlp_dict_out = nlp_dict_out

    def make_step_optimizer(self):
        arg = self.optimizer.arg
        U_offset = self.optimizer.nlp_dict_out['U_offset']
        # Extract the optimal control input to be applied
        nu = len(self.optimizer.u_mpc)

        step_index = int(self.simulator.t0_sim / self.simulator.t_step_simulator)
		# open window during the night is forbiddden
        for i in range(U_offset.shape[0]):
			# window uppper bound is at postion 2 -> +2
            arg['ubx'][U_offset[i][0] + 2] = self.optimizer.tv_p_values[step_index][-3,i]

        result = self.optimizer.solver(x0=arg['x0'], lbx=arg['lbx'], ubx=arg['ubx'], lbg=arg['lbg'], ubg=arg['ubg'], p = arg['p'])
        # Store the full solution
        self.optimizer.opt_result_step = data_do_mpc.opt_result(result)

        # nu = len(self.optimizer.u_mpc)
        # U_offset = self.optimizer.nlp_dict_out['U_offset']
        v_opt = self.optimizer.opt_result_step.optimal_solution
        self.optimizer.u_mpc = NP.resize(NP.array(v_opt[U_offset[0][0]:U_offset[0][0]+nu]),(nu))
    def make_step_observer(self):
        self.make_measurement()
        self.observer.observed_states = self.simulator.measurement # NOTE: this is a dummy observer

    def make_step_simulator(self):
        # Extract the necessary information for the simulation
        u_mpc = self.optimizer.u_mpc
        # Use the real parameters
        p_real = self.simulator.p_real_now(self.simulator.t0_sim)
        tv_p_real = self.simulator.tv_p_real_now(self.simulator.t0_sim)
        if self.optimizer.state_discretization == 'discrete-time':
            rhs_unscaled = substitute(self.model.rhs, self.model.x, self.model.x * self.model.ocp.x_scaling)/self.model.ocp.x_scaling
            rhs_unscaled = substitute(rhs_unscaled, self.model.u, self.model.u * self.model.ocp.u_scaling)
            rhs_fcn = Function('rhs_fcn',[self.model.x,vertcat(self.model.u,self.model.tv_p)],[rhs_unscaled])
            x_next = rhs_fcn(self.simulator.x0_sim,vertcat(u_mpc,tv_p_real))
            self.simulator.xf_sim = NP.squeeze(NP.array(x_next))
        else:
        # Get sizes of the variables
            nx = self.model.x.size(1)
            nz = self.model.z.size(1)
            result  = self.simulator.simulator(x0 = self.simulator.x0_sim[0:nx-nz], z0 = self.simulator.x0_sim[nx-nz:nx], p = vertcat(u_mpc,p_real,tv_p_real))
            self.simulator.xf_sim = NP.squeeze(vertcat(NP.squeeze(result['xf']),NP.squeeze(result['zf'])))
        # Update the initial condition for the next iteration
        self.simulator.x0_sim = self.simulator.xf_sim
        # Correction for sizes of arrays when dimension is 1
        if self.simulator.xf_sim.shape ==  ():
            self.simulator.xf_sim = NP.array([self.simulator.xf_sim])
        # Update the mpc iteration index and the time
        self.simulator.mpc_iteration = self.simulator.mpc_iteration + 1
        self.simulator.t0_sim = self.simulator.tf_sim
        self.simulator.tf_sim = self.simulator.tf_sim + self.simulator.t_step_simulator

    def step_simulator(self, model_fmu, simTime, secStep):

        lbub = NP.load('Neural_Network\lbub.npy')
        x_lb_NN = lbub[0]
        x_ub_NN = lbub[1]
        y_lb_NN = lbub[2]
        y_ub_NN = lbub[3]

        secStep = secStep*60
        simTime = simTime*60
        states = []

        u_mpc = self.optimizer.u_mpc
        tv_p_real = self.simulator.tv_p_real_now(self.simulator.t0_sim)
        rhs_unscaled = substitute(self.model.rhs, self.model.x, self.model.x * self.model.ocp.x_scaling)/self.model.ocp.x_scaling
        rhs_unscaled = substitute(rhs_unscaled, self.model.u, self.model.u * self.model.ocp.u_scaling)
        rhs_fcn = Function('rhs_fcn',[self.model.x,vertcat(self.model.u,self.model.tv_p)],[rhs_unscaled])
        x_next = rhs_fcn(self.simulator.x0_sim,vertcat(u_mpc,tv_p_real))


        # print "u_mpc: " + str(u_mpc)
        tmp = NP.copy(u_mpc)
        """
        Blinds
        """
        tmp[0:2] = NP.round(tmp[0:2])
        model_fmu.set('u_blinds_N', tmp[0])
        model_fmu.set('u_blinds_W', tmp[1])

        """
        AHU
        """
        u_mpc[2] = NP.floor(u_mpc[2]*10)/10
        model_fmu.set('u_AHU1_noERC', u_mpc[2])


        """
        Baseboard Heater
        """
        model_fmu.set('u_rad_OfficesZ1', u_mpc[3])

        # do one step simulation
        res = model_fmu.do_step(current_t=simTime, step_size=secStep, new_step=True)

        HeatRate = NP.zeros((1,1))
        for i in range(0, HeatRate.shape[0]):
            HeatRate[i,:] = model_fmu.get('HeatRate_Z'+str(i+1))
        # Concatenate past HeatRate values with present ones
        self.simulator.HeatRate = NP.concatenate((self.simulator.HeatRate, HeatRate), axis = 1)


        if u_mpc[2] > 0 and HeatRate > 0:
            self.simulator.faultDetector = NP.concatenate((self.simulator.faultDetector, NP.ones((1,1)) ), axis = 1)
        else:
            self.simulator.faultDetector = NP.concatenate((self.simulator.faultDetector, NP.zeros((1,1)) ), axis = 1)

        # Get the new states
        states.append(model_fmu.get('Tzone_1'))
        # states.append(model_fmu.get('HeatRate_Z1')/100)
        states.append(model_fmu.get('u_blinds_N_val'))
        states.append(model_fmu.get('u_blinds_W_val'))
        states.append(model_fmu.get('u_AHU1_noERC_SchedVal'))
        states.append(model_fmu.get('SchedVal_Z1'))
        states.append(model_fmu.get('v_IG_Offices'))
        states.append(model_fmu.get('v_Tamb'))
        states.append(model_fmu.get('v_solGlobFac_E'))
        states.append(model_fmu.get('v_solGlobFac_N'))
        states.append(model_fmu.get('v_solGlobFac_S'))
        states.append(model_fmu.get('v_solGlobFac_W'))
        states.append(model_fmu.get('windspeed_Z1'))
        states.append(model_fmu.get('Hum_amb'))
        # states.append(model_fmu.get('Hum_zone'))
        states.append(model_fmu.get('P_amb'))
        states = NP.squeeze(states)


        # keep track of unmetHours
        diff = NP.zeros((1,1))
        step_index = int(self.simulator.t0_sim / self.simulator.t_step_simulator)
        if states[0] < self.optimizer.tv_p_values[step_index,-1,0]:
            diff = NP.resize(NP.abs(states[0] - self.optimizer.tv_p_values[step_index,-1,0]),(1,1))
        elif states[0] >  self.optimizer.tv_p_values[step_index,-2,0]:
            diff = NP.resize(NP.abs(states[0] - self.optimizer.tv_p_values[step_index,-2,0]),(1,1))
        self.simulator.unmetHours = NP.concatenate((self.simulator.unmetHours, diff), axis = 1)



        disturbances = NP.load('Neural_Network\disturbances.npy').squeeze()
        disturbances_lb = NP.min(disturbances,axis =1)
        disturbances_ub = NP.max(disturbances,axis =1)
        x_lb = NP.concatenate((x_lb_NN[0:5] - 1e-2, disturbances_lb)) # -20000* NP.ones(features*numbers)#
        x_ub = NP.concatenate((x_ub_NN[0:5] + 1e-2, disturbances_ub)) # 20000* NP.ones(features*numbers)#
        if (NP.squeeze(NP.asarray(x_next)) - x_lb < 0).all() or (x_ub - NP.squeeze(NP.asarray(x_next))  < 0).all():
            bp()
        # print states - x_next
        self.simulator.xf_sim = states
        # Update the initial condition for the next iteration
        self.simulator.x0_sim = self.simulator.xf_sim
        # Correction for sizes of arrays when dimension is 1
        if self.simulator.xf_sim.shape ==  ():
            self.simulator.xf_sim = NP.array([self.simulator.xf_sim])
        # Update the mpc iteration index and the time
        self.simulator.mpc_iteration = self.simulator.mpc_iteration + 1
        self.simulator.t0_sim = self.simulator.tf_sim
        self.simulator.tf_sim = self.simulator.tf_sim + self.simulator.t_step_simulator

    def f_const(self,var):
        if var <= 0*100:
            return 17
        elif var <= 1*100:
            return 22
        elif var <= 2*100:
            return 23
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

    def f_const_ahu(self,var):
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

    def compare(self, model_fmu, simTime, secStep):
        lbub = NP.load('Neural_Network\lbub.npy')
        x_lb_NN = lbub[0]
        x_ub_NN = lbub[1]
        y_lb_NN = lbub[2]
        y_ub_NN = lbub[3]
        secStep = secStep*60
        simTime = simTime*60
        duration = 144*10
        result_NN = NP.zeros((duration,self.model.x.shape[0]))
        result_Ep = NP.zeros((duration,self.model.x.shape[0]))
        Heatrate = []
        sched = NP.zeros((duration,4))
        for i in range(0,duration):
            u_mpc = self.optimizer.u_mpc
            u_mpc = NP.asarray(u_mpc, dtype=np.float32)
            u_mpc[3] = float(self.f_const(i)) #np.round(np.random.normal(17,0.1,1),2)
            tv_p_real = self.simulator.tv_p_real_now(self.simulator.t0_sim)
            rhs_unscaled = substitute(self.model.rhs, self.model.x, self.model.x * self.model.ocp.x_scaling)/self.model.ocp.x_scaling
            rhs_unscaled = substitute(rhs_unscaled, self.model.u, self.model.u * self.model.ocp.u_scaling)
            rhs_fcn = Function('rhs_fcn',[self.model.x,vertcat(self.model.u,self.model.tv_p)],[rhs_unscaled])
            x_next = rhs_fcn(self.simulator.x0_sim,vertcat(u_mpc,tv_p_real))
            self.simulator.xf_sim = NP.squeeze(NP.array(x_next))
            self.simulator.x0_sim = self.simulator.xf_sim

            self.simulator.mpc_iteration = self.simulator.mpc_iteration + 1
            self.simulator.t0_sim = self.simulator.tf_sim
            self.simulator.tf_sim = self.simulator.tf_sim + self.simulator.t_step_simulator


            result_NN[i,:] = self.simulator.xf_sim
        for i in range(0,duration):
            states = []
            u_mpc = self.optimizer.u_mpc
            u_mpc = NP.asarray(u_mpc, dtype=np.float32)
            u_mpc[3] = float(self.f_const(i)) #np.round(np.random.normal(17,0.1,1),2)
            """
            Blinds
            """
            model_fmu.set('u_blinds_N', u_mpc[0])
            model_fmu.set('u_blinds_W', u_mpc[1])


            """
            AHU
            """
            model_fmu.set('u_AHU1_noERC', u_mpc[2])

            # model_fmu.set('u_AHU2_noERC', u_mpc[4])

            """
            Baseboard Heater
            """
            model_fmu.set('u_rad_OfficesZ1', u_mpc[3])
            # do one step simulation
            res = model_fmu.do_step(current_t=simTime, step_size=secStep, new_step=True)

            # Get the new states
            states.append(model_fmu.get('Tzone_1'))
            # states.append(model_fmu.get('HeatRate_Z1')/100)
            Heatrate.append(model_fmu.get('HeatRate_Z1')/63.15)
            states.append(model_fmu.get('u_blinds_N_val'))
            states.append(model_fmu.get('u_blinds_W_val'))
            states.append(model_fmu.get('u_AHU1_noERC_SchedVal'))
            states.append(model_fmu.get('SchedVal_Z1'))
            states.append(model_fmu.get('v_IG_Offices'))
            states.append(model_fmu.get('v_Tamb'))
            states.append(model_fmu.get('v_solGlobFac_E'))
            states.append(model_fmu.get('v_solGlobFac_N'))
            states.append(model_fmu.get('v_solGlobFac_S'))
            states.append(model_fmu.get('v_solGlobFac_W'))
            states.append(model_fmu.get('windspeed_Z1'))
            states.append(model_fmu.get('Hum_amb'))
            states.append(model_fmu.get('P_amb'))

            states = NP.squeeze(states)

            simTime += secStep

            result_Ep[i,:] = states
            sched[i,:] = u_mpc


        rmse = NP.sqrt(NP.mean((result_Ep[:,0]-result_NN[:,features*(numbers-1)])**2,axis=0))
        bp()
        NP.save('open_loop_ANN.npy', NP.concatenate((result_Ep, result_NN, sched),axis = 1))
        NP.save('Heatrate.npy', Heatrate)
        # NP.sqrt(NP.mean((result_Ep - result_NN)**2,axis=0))
        print("RMSE: " + str(rmse))
        NP.argmax(NP.abs(result_NN[:,features*(numbers-1)]-result_Ep[:,0]),axis=0)
        P.subplot(3, 1, 1)
        P.plot(result_NN[:,0], linewidth = 2.0)
        P.plot(result_Ep[:,0], linewidth = 2.0)
        P.plot(sched[:,-1])
        P.ylabel('ZoneTemp')
        P.legend(['NN','E+','Setpoint'])
        P.subplot(3, 1, 2)
        P.plot(result_NN[:,features*(numbers-1)] - result_Ep[:,0])
        # P.plot(result_NN[:,8] - result_Ep[:,8])
        P.ylabel('Difference')

        P.subplot(3, 1, 3)
        P.plot(result_NN[:,features*(numbers-1) + 7])
        P.plot(result_Ep[:,7])
        P.ylabel('Difference v_IG_Offices')
        P.show()



    def make_measurement(self):
        # NOTE: Here implement the own measurement function (or load it)
        # This is a dummy measurement
        self.simulator.measurement = self.simulator.xf_sim

    def prepare_next_iter(self):
        observed_states = self.observer.observed_states
        X_offset = self.optimizer.nlp_dict_out['X_offset']
        nx = self.model.ocp.x0.size(1)
        nu = self.model.u.size(1)
        ntv_p = self.model.tv_p.size(1)
        nk = self.optimizer.n_horizon
        parameters_setup_nlp = struct_symMX([entry("uk_prev",shape=(nu)), entry("TV_P",shape=(ntv_p,nk))])
        param = parameters_setup_nlp(0)
        # First value of the nlp parameters
        param["uk_prev"] = self.optimizer.u_mpc
        step_index = int(self.simulator.t0_sim / self.simulator.t_step_simulator)
        param["TV_P"] = self.optimizer.tv_p_values[step_index]
        # print "tv_p: " + str(param["TV_P"][:,:])
        self.optimizer.arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = observed_states
        self.optimizer.arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = observed_states
        self.optimizer.arg["x0"] = self.optimizer.opt_result_step.optimal_solution
        # Pass as parameter the used control input
        self.optimizer.arg['p'] = param

    def store_mpc_data(self):
        mpc_iteration = self.simulator.mpc_iteration - 1 #Because already increased in the simulator
        data = self.mpc_data
        #pdb.set_trace()
        step_index = int( (self.simulator.t0_sim - self.simulator.t_step_simulator) / self.simulator.t_step_simulator)
        tv_p = self.optimizer.tv_p_values[step_index]
        data.mpc_tv_p = NP.append(data.mpc_tv_p, [tv_p[:,0]], axis = 0)
        data.mpc_states = NP.append(data.mpc_states, [self.simulator.xf_sim], axis = 0)
        #pdb.set_trace()
        data.mpc_control = NP.append(data.mpc_control, [self.optimizer.u_mpc], axis = 0)
        #data.mpc_alg = NP.append(data.mpc_alg, [NP.zeros(NP.size(self.model.z))], axis = 0) # TODO: To be completed for DAEs
        data.mpc_time = NP.append(data.mpc_time, [[self.simulator.t0_sim]], axis = 0)
        data.mpc_cost = NP.append(data.mpc_cost, self.optimizer.opt_result_step.optimal_cost, axis = 0)
        #data.mpc_ref = NP.append(data.mpc_ref, [[0]], axis = 0) # TODO: To be completed
        stats = self.optimizer.solver.stats()
        data.mpc_cpu = NP.append(data.mpc_cpu, [[stats['t_wall_mainloop']]], axis = 0)
        data.mpc_parameters = NP.append(data.mpc_parameters, [self.simulator.p_real_now(self.simulator.t0_sim)], axis = 0)
