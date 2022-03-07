import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate

import machine
import sequencing
import job_creation
import breakdown_creation
import heterogeneity_creation

class shopfloor:
    def __init__(self,env,span,m_no,utl,**kwargs):
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
        (self.env, self.span, self.m_list, [1,50], 3, utl, print = 0)
        #self.job_creator.initial_output()

        # STEP 4: initialize all machines
        for i in range(m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machines
            exec(expr3)

        # STEP 5: check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                sqc_expr = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(sqc_expr)
                except:
                    print("WARNING: Rule assigned is invalid !")
                    raise Exception

    # FINAL STEP: start the simulaiton
    def simulation(self):
        self.env.run()

# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}
# list of experiments
m_no = 10
span = 2000
iteration = 20

utl=[0.7,0.8,0.9]
# create the figure instance
fig = plt.figure(figsize=(10,13),)
panel_titles=['(a) 70% utilization rate','(b) 80% utilization rate','(c) 90% utilization rate']

for i,utilization in enumerate(utl):
    for idx in range(iteration):
        env = simpy.Environment()
        spf_dict[idx] = shopfloor(env, span, m_no, utl[i])
        spf_dict[idx].simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf_dict[idx].job_creator.tardiness_output()
    # lower half, information of jobs
    ax = fig.add_subplot(3,1,i+1)
    ax.set_ylabel('Number of jobs in system')
    y_main_range = []
    x=[]
    y=[]
    for idx in range(iteration):
        arrival, departure, in_system = spf_dict[idx].job_creator.timing_output()
        # plot the number of in-system jobs
        xs = list(in_system.keys())
        xs.insert(0,0)
        xs.pop()
        ys = list(in_system.values())
        y_main_range.append(np.max(ys))
        x+=xs
        y+=ys
    #print(x,y)
    data = np.array([x,y]).transpose()
    #print(data)
    data = data[data[:, 0].argsort()].transpose()
    #print(data)
    x = data[0]
    y = data[1]
    z = np.polyfit(x, y, deg=3)
    p = np.poly1d(z)
    y_est = p(x)
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    ax.plot(x, y_est, '-', lw=2,label='Polynomial fit')
    ax.fill_between(x, y_est - y_err/5, y_est + y_err/5, alpha=0.2)
    ax.scatter(x, y, s=2,color='tab:brown', alpha=0.5,zorder=3,label='Number of jobs')
    y_main_range = np.ceil(np.max(y_main_range)/5)*5
    # ticks
    fig_major_ticks = np.arange(0, span+1, span/10)
    ax.set_xticks(fig_major_ticks)
    ax.set_yticks(np.arange(0,y_main_range+5,5))
# different settings for the grids:
    ax.grid(axis='x')
    ax.grid(axis='y')
    ax.set_xlim(0,span)
    ax.set_ylim(0,y_main_range)
    ax.set_title(panel_titles[i])
    ax.legend()
ax.set_xlabel('Time of simulation')

fig.subplots_adjust(top=0.9, bottom=0.1, right=0.9, wspace=0.25, hspace=0.28)
#fig.savefig(sys.path[0]+"/Thesis_figures/job_in_system.png", dpi=400, bbox_inches='tight')
plt.show()
