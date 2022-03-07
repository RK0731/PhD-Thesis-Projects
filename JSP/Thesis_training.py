import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import numpy as np

import machine
import sequencing
import job_creation
import breakdown_creation
import heterogeneity_creation
import brain
import validation

class shopfloor:
    def __init__(self,env,span,m_no,**kwargs):
        # STEP 1: create environment for simulation and control parameters
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []

        # STEP 2: create instances of machines
        for i in range(m_no):
            expr1 = '''self.m_{} = machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)

        # STEP 3: initialize the initial jobs, distribute jobs to workcenters
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz, print
        self.job_creator = job_creation.creation\
        (self.env, self.span, self.m_list, [1,50], 3, 0.8, print = 0)
        self.job_creator.initial_output()

        # STEP 4: initialize all machines
        for i in range(m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machines
            exec(expr3)

        #STEP 5: add a brain to the shop floor
        self.brain = brain.brain(self.env, self.job_creator, self.m_list, self.span/10, self.span,\
        DDQN_SI = 0, TEST = 0, A2C = 0, IQL = 0, I_DDQN = 0, \
        AS = 0, TEST_AS = 0, IQL_AS = 0, I_DDQN_AS = 1, \
        expert = 0, bsf_start = 0, reward_function = 12)


    def training_record(self, m_idx):
        fig,(ax0,ax1) = plt.subplots(2,1, figsize=(10,6), sharex=True)
        ax0.scatter(np.array(self.brain.loss_time_record), np.array(self.brain.loss_record), s=3, c='b')
        ax1.scatter(np.array(self.job_creator.reward_record[m_idx][0]), np.array(self.job_creator.reward_record[m_idx][1]), s=3, c='r')
        ax0.set_ylabel('Loss of training')
        ax1.set_ylabel('Reward')
        ax1.set_xlabel('Time')
        ax0.grid()
        ax1.grid()
        plt.show()

    # FINAL STEP: start the simulaiton, and plot the loss/ reward record
    def simulation(self):
        self.env.run()
        self.brain.check_parameter()
        self.training_record(1)

# create the environment instance for simulation
env = simpy.Environment()
span = 20000
scale = 10
show = True
# create the shop floor instance
# the command of startig the simulation is included in shopfloor instance, run till there's no more events
spf = shopfloor(env, span, scale,)
spf.simulation()
# information of job sequence, processing time, creation, due and tardiness
#spf.job_creator.final_output()
if show:
    for m in spf.m_list:
        print(m.cumulative_run_time/span)
