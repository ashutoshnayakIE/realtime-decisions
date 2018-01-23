import random
from gurobipy import*
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import ExcelWriter

import class_file
import function_file
import global_file

# setting up the prices for the energy cost
price_t = [18.73,18.01,17.7,18.28,19.38,21.06,25.61,24.18,23.21,22.78,22.13,21.58,21.12,20.69,20.38,21.23,25.91,27.68,25.44,24.36,22.56,21.06,19.91,20.16]
# price_nr is the price for non - renewable energy
price_nr = []
for t in range(24):
    price_nr.append(price_t[t])
    price_nr.append(price_t[t])

price_nr = np.array([price_nr]*30*2)
price_nr = price_nr.flatten()

# case_priority = 1: based on arrival for FCFS
# case_priority = 2: based on earliest due date EDD
# case_priority = 3: based on Shortest processing time SPT
# case_priority = 4: based on critical ratio CR

# case_demand = 1: jobs arrive at the start of the day
# case_demand = 2: jobs arrive throughout the day

# scheduling_model = 1 is for non peak cases
# scheduling_model = 2 is for the peak model
# scheduling_model = 3 is for non peak model with prediction
# scheduling_model = 4 is for the peak model with prediction
# scheduling model = 5 is Lyapunov model
# scheduling model = 6 is Lyapunov model with prediction

# in model 3 and 4, we decide should we turn off the machine and turn it on later
# two things we check:
#     1) future site - renewable < current
#     2) switching off and on is beneficial or not (time period should be more).

# M machines
# N job types
# t_dash is looking at three hours ahead

wind_multiply = 5

global_file.M = 5
global_file.N = 5
AM = 15#*wind_multiply/2.0
YM = 3#*wind_multiply/2.0
BM = 3#*wind_multiply/2.0
global_file.on_machine = 12
global_file.off_machine = 6
global_file.idle_machine = 4
global_file.stay_time = 6

# equation for wind generation from the wind mill (in MW)
# SARIMA(4,0,0)x(1,0,0) model is used

windPower = pd.read_csv("dataPower.csv")['windPower'].tolist()
site = pd.read_csv("site.csv")['site'].tolist()
global_file.site = np.divide(np.array(site),20)

# process generation
# tells us about how many processes each part has and ....
# what machines are eligible for each job

# read as part type, machine, time , power
global_file.num_steps = 5
global_file.time_steps = 4
process = []

for i in range(global_file.N):
    process.append([])
    numberProcess = random.randint(1,global_file.num_steps)
    processing_time = 0
    
    for k in range(numberProcess):
        machine = random.randint(0,global_file.M-1)
        time = random.randint(1,global_file.time_steps)
        processing_time += time
        power = random.randint(20,55)
        process[i].append([machine,time,power])
    process[i].insert(0,[numberProcess,processing_time])

# rate of arrival is reduced by half as shift is only half the day
# shift starts at 7 am and ends at 7 pm
global_file.shift_start = 7
global_file.shift_end = 19  
   
# updating the global variables

global_file.process = process

# DIFFERENT MODELS ------------------------------------------------------------------------------

# first model is scheduling_model = 1
# second model is scheduling_model = 2
# 1) FCFS
# 2) FCFS+peak
# 3) SPT
# 4) SPT+peak 
# 5) EDD
# 6) EDD+peak
# 7) CR
# 8) CR+peak 

# we can change the time horizon to understand the performance of the algorithm
# it depends on the value of t_dash

# delay threshold is set for model 3 and model 4
# if the ratio of idel_power/time delayed is less than this threshold, do not turn on the machine

case_p = [1,2,3,4]
scheduling_model = [2]
peak = [120,150,200,250]
delay = [0.1,1,2,5]
delay_l = [0,3,5,10]
t_d = [1,3,5,7]
replications = 30
model = -1

# number of models
nom = 16
result = np.zeros([nom,replications,7])     # 8 models, 30 replications, 6 measures
result_print = np.zeros([nom,7])

resultOutput = ExcelWriter('results1.xlsx')
toprint = 1
t_dash = 3

for p in range(4):
    for td in range(4):
        model += 1
        random_seed = 0
        random.seed(random_seed) 

        case_priority = 1#case_p[cp]

        scheduling_model = 5 
        # case priority is reset for lyapunov model
        if scheduling_model == 5 or scheduling_model == 6:
            case_priority = 4
            
        peak_constraint = peak[p]
        delay_threshold1 = delay[1]
        
        # these are the threshold values for the two variables
        delay_lyapunov = delay_l[td]
        t_dash = 3#t_d[td]
        peak_lyapunov = 0
        peak_constraint_lyapunov = peak_constraint - 20
        
        case_demand = 2    
        if case_demand == 1:        
            jobs_arrival_rate = (float(global_file.M)*24)/(24*global_file.num_steps*global_file.time_steps*global_file.N)
        else:
            jobs_arrival_rate = (float(global_file.M)*24)/(2*24*global_file.num_steps*global_file.time_steps*global_file.N)
        
        for reps in range(replications):
            arrival = 0
            throughput = 0
            operation = 0
            dueDate = 0
            proportion_completed = 0
            
            # the following variables should collect the value of the constraints
            D = 0
            P = 0
            alpha_p = 0.1
            alpha_d = 0.1
            mu_peak = 1
            
            peak_power = np.array([0]*24*2*30*2,dtype=float)  # 24 hours, 30 minute one hour, and 30 days, 2 months
            
            PE_array = [] 
            battery_array = []          
            
            A = 0
            
            # initializing machines
            machines = []
            for j in range(global_file.M):
                machines.append(class_file.machine(j,[],0,0,global_file.on_machine,global_file.off_machine,global_file.idle_machine,0,0))
            
            # 5 days are warm-up, rest is what is noted
            for t in range(96,96+48*5+48*30):
                
                # outside the shift, just consider product demand
                
                if (t/2)%24 < global_file.shift_start or (t/2)%24 > global_file.shift_end:
                    
                    peak_power[t] += random.normalvariate(global_file.site[(t/2)%24],3)
                    
                    for j in range(global_file.M):
                        if machines[j].on_off == 1:
                            machines[j].on_off = 0
                            peak_power[t] += machines[j].off_power
                        
                    
                    # job types entering in each time slot
                    new_jobs = np.random.poisson(jobs_arrival_rate, global_file.N)
                    ts = (t/48)*48
                    
                    for j1 in range(global_file.N):
                        for j2 in range(new_jobs[j1]):
                            if t > 96+48*5:
                                arrival += 1
                            new_part = class_file.part(j1,t,0,ts,process[j1][0][1]+(int(np.random.uniform(1,6.5)))*48)#np.random.uniform(1,6.5)
                            machine = new_part.machine
                            new_part.start = t
                            new_part.due_date += t
                            machines[machine].queue = function_file.queue_sorting(machines[machine].queue,new_part,case_priority,t)
                   
                    # 100 to convert to KW because the original data is in MW/10    
                    A += windPower[t/2]*wind_multiply
                    A = min(AM,A)
                    
                # inside the shift, the optimization decisions are made
                else:
                    ts = (t/48)*48
                    # new jobs are added only in case 2 of dynamic arrivals
                    if case_demand == 2:
                        new_jobs = np.random.poisson(jobs_arrival_rate, global_file.N)
                        for j1 in range(global_file.N):
                            for j2 in range(new_jobs[j1]):
                                if t > 96+48*5:
                                    arrival += 1
                                new_part = class_file.part(j1,t,0,ts,process[j1][0][1]+(int(np.random.uniform(1,6.5)))*48)
                                machine = new_part.machine
                                new_part.start = t
                                new_part.due_date += t
                                machines[machine].queue = function_file.queue_sorting(machines[machine].queue,new_part,case_priority,t)
                    
                    # non-prediction model
                    if scheduling_model == 1 or scheduling_model == 2:
                        
                        peak_power[t] += random.normalvariate(global_file.site[(t/2)%24],3)
                    
                        A += windPower[t/2]*wind_multiply
                        peak_power[t] -= min(YM,A)
                        A -= min(YM,A)
                        A = min(AM,A)
                    
                        for j in range(global_file.M):
                            
                            # turning on the machines
                            if machines[j].on_off == 0:
                                machines[j].on_off = 1
                                peak_power[t] += global_file.on_machine
                            
                            if len(machines[j].queue) == 0:
                                peak_power[t] += global_file.idle_machine
                            
                            else:
                                q_to_do = 0
                                flag_peak = 0
                                if scheduling_model == 2:
                                    for q in range(len(machines[j].queue)):
                                        current_part = machines[j].queue[0]
                                        time = current_part.time
                                        power = current_part.power
                                        if peak_power[t]+power <= peak_constraint:
                                            machines[j].scheduled_time = max(t,machines[j].scheduled_time)
                                            flag_peak = 1
                                            q_to_do = q
                                            break
                                
                                # stopping the machines if shift is ending
                                if ((machines[j].scheduled_time + time)/2)%24 > global_file.shift_end:
                                    machines[j].on_off = 0
                                    peak_power[machines[j].scheduled_time] += global_file.off_machine
                                    
                                # scheduling jobs if not preoccupied
                                elif machines[j].scheduled_time <= t and flag_peak == 1 and current_part.min_start_time <= t:
                                    peak_power[machines[j].scheduled_time:machines[j].scheduled_time+time] += power
                                    machines[j].scheduled_time += time
                                    
                                    # updating the attributes of the part
                                    current_part = function_file.update_part(current_part,machines[j].scheduled_time)
                                    if t > 96+48*5:
                                        operation += 1
                                    
                                    del machines[j].queue[q_to_do]
                                    # moving the job to the next machine
                                    if current_part.state < current_part.total_task:
                                        machines[current_part.machine].queue = function_file.queue_sorting(machines[current_part.machine].queue,current_part,case_priority,t)
                                    else:
                                        if t > 96+48*5:
                                            throughput += 1
                                            dueDate += max(0,machines[j].scheduled_time-current_part.due_date)
                                        
                    # Prediction model  
                    elif scheduling_model == 3 or scheduling_model == 4:
                        
                        peak_power[t] += random.normalvariate(global_file.site[(t/2)%24],3)
                        
                        # performing actions during the shift    
                        prediction = function_file.energy(t/2,t_dash,wind_multiply,windPower)
                        power_prediction = np.array(prediction[0])
                        site_prediction = np.array(prediction[1])
                        A += windPower[t/2]*wind_multiply
                        
                        for j in range(global_file.M):
                            
                            flag = 0
                            min_t = min(site_prediction-power_prediction)
                            min_t_index = np.argmin(site_prediction-power_prediction)

                            # making the decision to switch on or off    
                            if machines[j].on_time + global_file.stay_time < t and global_file.site[(t/2)%24]-windPower[t/2]*wind_multiply- min_t+machines[j].idle_power/(machines[j].on_power+machines[j].off_power) > delay_threshold1 and global_file.site[(t/2)%24]-windPower[t/2]*wind_multiply > min_t:
                                machines[j].on_time = max(t,machines[j].on_time)+1+min_t_index
                                site_prediction[min_t_index] += machines[j].on_power
                                
                                # turning off if it is on or keep it idle
                                # first condition is to keep the machine off for at least 3 hours
                                if machines[j].on_off == 1 and machines[j].idle_power* float(1+min_t_index) > machines[j].on_power + machines[j].off_power:
                                    peak_power[t] += machines[j].off_power
                                    machines[j].on_off = 0
                                    machines[j].off_time = t
                             
                            else:
                                machines[j].on_time = t
                                if machines[j].on_off == 0:
                                    peak_power[t] += machines[j].on_power
                                    machines[j].on_off = 1
                            
                            if len(machines[j].queue) > 0 and machines[j].on_off == 1 and t >= machines[j].on_time:
                                
                                q_to_do = 0
                                flag_peak = 0
                                if scheduling_model == 4:
                                    for q in range(len(machines[j].queue)):
                                        current_part = machines[j].queue[0]
                                        time = current_part.time
                                        power = current_part.power
                                        if peak_power[t]+power <= peak_constraint:
                                            machines[j].scheduled_time = max(t,machines[j].scheduled_time)
                                            flag_peak = 1
                                            q_to_do = q
                                            break
                                
                                # stopping the machines if shift is ending
                                if ((machines[j].scheduled_time + time)/2)%24 > global_file.shift_end:
                                    machines[j].on_off = 0
                                    peak_power[machines[j].scheduled_time] += machines[j].off_power
                                    
                                # scheduling jobs if not preoccupied
                                elif machines[j].scheduled_time <= t and flag_peak == 1 and current_part.min_start_time <= t:
                                    peak_power[machines[j].scheduled_time:machines[j].scheduled_time+time] += power
                                    machines[j].scheduled_time += time
                                    
                                    # updating the attributes of the part
                                    current_part = function_file.update_part(current_part,machines[j].scheduled_time)
                                    if t > 96+48*5:
                                        operation += 1
                                    
                                    del machines[j].queue[q_to_do]
                                    # moving the job to the next machine
                                    if current_part.state < current_part.total_task:
                                        machines[current_part.machine].queue = function_file.queue_sorting(machines[current_part.machine].queue,current_part,case_priority,t)
                                    else:
                                        proportion_completed = throughput/float(0.001+arrival)
                                        if t > 96+48*5:
                                            throughput += 1
                                            dueDate += max(0,machines[j].scheduled_time-current_part.due_date)
                        
                        A = min(AM,A)
                    
                    elif scheduling_model == 5 or scheduling_model == 6:
                        
                        optimal = Model("OP")
                        optimal.setParam('OutputFlag', False)
                        
                        # performing actions during the shift    
                        prediction = function_file.energy(t/2,t_dash,wind_multiply,windPower)
                        power_prediction = np.array(prediction[0])
                        site_prediction = np.array(prediction[1])
                        
                        siteC = random.normalvariate(global_file.site[(t/2)%24],1);
                        reneP = windPower[t/2]*wind_multiply
                        
                        peak_power[t] += siteC
                        
                        A = max(0,A)
                        A = min(A,AM)
                        
                        x = [[] for j in range(global_file.M)]
                        # y is for the machines to be turned on 
                        # z is for the machines to be turned off
                        # v is for the machines to be idle
                        y = [optimal.addVar(0,1,0,GRB.BINARY,'y'+str(j)) for j in range(global_file.M)]
                        z = [optimal.addVar(0,1,0,GRB.BINARY,'y'+str(j)) for j in range(global_file.M)]
                        v = [optimal.addVar(0,1,0,GRB.BINARY,'v'+str(j)) for j in range(global_file.M)]
                        y_active = [0]*global_file.M
                        
                        X = [optimal.addVar(0,GRB.INFINITY,0,GRB.CONTINUOUS,'X') for l in range(t_dash*2+1)]
                        # from renewables at time t
                        R = optimal.addVar(0,reneP,0,GRB.CONTINUOUS,'R')
                        # measures the load of all the time periods
                        # +1 is added to account for the current time period
                        L = [optimal.addVar(0,GRB.INFINITY,0,GRB.CONTINUOUS,'L') for l in range(t_dash*2+1)] 
                        load = [LinExpr() for l in range(t_dash*2+1)] 
                        
                        B = optimal.addVar(0,min(BM,A),0,GRB.CONTINUOUS,'B')
                        Y = optimal.addVar(0,min(max(0,A-YM),YM),0,GRB.CONTINUOUS,'Y')
                        # variable for maintaining the Peak
                        PE = optimal.addVar(0,GRB.INFINITY,0,GRB.CONTINUOUS,'PE')
                        
                        # due date reduction is observed with respect to CR basis
                        # therefore when a job joins in a machine, it is queued based on CR
                        dd_reduction = 0
                        
                        objective = LinExpr()
                        
                        objective.addTerms(alpha_p*P,PE)
                        
                        # this is for current step in lyapunov
                        objective.addTerms(mu_peak,PE) 
                        
                        optimal.update()
                        
                        # these are for asking to choose from renewables
                        
                        objective.addTerms(price_nr[t]/1000.0,X[0])
                        objective.addTerms(price_nr[t]/2000.0,R)
                        objective.addTerms(price_nr[t]/2000.0,B)
                        objective.addTerms(-price_nr[t]/4000.0,B)
                        
                        qLength = [len(machines[j].queue) for j in range(global_file.M)]
                        
                        for j in range(global_file.M):
                            if t < machines[j].off_time or t < machines[j].scheduled_time:
                                y1 = LinExpr()
                                y1.addTerms(1,y[j])
                                y1.addTerms(1,z[j])
                                y1.addTerms(1,v[j])
                                optimal.addConstr(y1,GRB.EQUAL,0)
                                   
                            if t >= machines[j].scheduled_time and t >= machines[j].off_time and len(machines[j].queue) > 0:
                                
                                # making variable = 0 if it satisfies a criteria
                                y_active[j] = 1
                                
                                temp_dd = 0
                                # turning on machine
                                if machines[j].on_off == 0:
                                    load[0].addTerms(machines[j].on_power, y[j])
                                    # the constraint that it cannot be continued or stopped
                                    optimal.addConstr(z[j],GRB.EQUAL,0)
                                   
                                elif machines[j].on_off == 1:
                                    optimal.addConstr(y[j],GRB.EQUAL,0)
                                    
                                    # either idle power consumption or off power consumption
                                    load[0].addTerms(machines[j].off_power, z[j])
                                    
                                    # machine is either on or off
                                    machine_on_off = LinExpr()
                                    machine_on_off.addTerms(1,z[j])
                                    machine_on_off.addTerms(1,v[j])
                                    optimal.addConstr(machine_on_off,GRB.EQUAL,1,'mof')
                                    
                                for q in range(len(machines[j].queue)):
                                    # x_decision accounts for x is schedule just once
                                    x_decision = LinExpr()
                                    temp_x = []
                                    
                                    # adding the dd lyapunov
                                    for l in range(t_dash*2+1):
                                        temp_x.append(optimal.addVar(0,1,0,GRB.BINARY,'x'+str(j)+str(q)))
                                        optimal.update()
                                        x_decision.addTerms(1,temp_x[-1])
                                        
                                        # adding priority among the jobs
                                        job_priority = min(len(machines[j].queue),t_dash*2)-q
                                        objective.addTerms(-alpha_d*D*job_priority,temp_x[-1])
                                        
                                        # adding the load for the L
                                        load[l].addTerms(machines[j].queue[q].power,temp_x[-1])
                                        
                                        # constraint that x <= y
                                        scheduling_condition = LinExpr()
                                        scheduling_condition.addTerms(1,temp_x[-1])
                                        scheduling_condition.addTerms(-1,y[j])
                                        scheduling_condition.addTerms(-1,v[j])
                                        optimal.addConstr(scheduling_condition,GRB.LESS_EQUAL,0,'xy')
                                        
                                        # constraint that it is not scheduled outside the shift
                                        if ((t+l)/2)%24 > global_file.shift_end:
                                            optimal.addConstr(temp_x[-1],GRB.EQUAL,0,'xy')
                                    
                                    # ensuring that it continues once it is started
                                    time = machines[j].queue[q].time
                                    for jj1 in range(t_dash*2+1-time):
                                        for jj2 in range(time-1):
                                            tcons = LinExpr()
                                            tcons.addTerms(1,temp_x[jj1])
                                            tcons.addTerms(-1,temp_x[jj1+jj2+1])
                                            optimal.addConstr(tcons,GRB.LESS_EQUAL,0,'tcons')
                                    
                                    
                                    # ensuring that it starts at only one point    
                                    optimal.addConstr(x_decision,GRB.LESS_EQUAL,machines[j].queue[q].time,'x_decn')
                                    x[j].append(temp_x)
                                    
                                    if temp_dd + machines[j].queue[q].time <= t_dash*2+1:
                                        # dd_reduction stores the sum of dd reduction if machine was on
                                        # the weight of a job is given as length of queue - q = priority values
                                        # since jobs are arranged according to the CR priority
                                        dd_reduction += min(len(machines[j].queue),t_dash*2)-q
                                        temp_dd += machines[j].queue[q].time
                             
                        for l in range(t_dash*2+1):
                            # adding the load for total consumption
                            objective.addTerms(price_nr[t]/1000.0,L[l])
                         
                        optimal.setObjective(objective, GRB.MINIMIZE)
                        
                        temp_peak = [max(peak_power[t],peak_constraint_lyapunov)]
                        for l in range(t_dash*2):
                            temp_peak.append(max(peak_constraint_lyapunov,peak_power[t+1+l]+site_prediction[l]-power_prediction[l]))
                        
                        # adding constraint for the loads 
                        for l in range(t_dash*2+1):
                            
                            for j in range(global_file.M):
                               
                                #setting machine simultaneous constraint
                                if y_active[j] != 0:
                                
                                    machine_simultaneous_constraint = LinExpr()
                                    for q in range(len(machines[j].queue)):
                                        machine_simultaneous_constraint.addTerms(1,x[j][q][l])
                                    optimal.addConstr(machine_simultaneous_constraint,GRB.LESS_EQUAL,1,'machineconstr')
                                    
                            load[l].addTerms(-1,L[l])
                            
                            supply_demand = LinExpr()
                            supply_demand.addTerms(-1,L[l])
                            supply_demand.addTerms(1,X[l])
                            
                            if l == 0:
                                # equation for equating the load at time t
                                # this is for the supply demand matching
                                supply_demand.addTerms(1,B)
                                supply_demand.addTerms(1,R)
                                supply_demand.addTerms(-1,Y) 
                                optimal.addConstr(load[l],GRB.EQUAL,-peak_power[t])
                                optimal.addConstr(supply_demand,GRB.EQUAL,0)
                            else:
                                optimal.addConstr(load[l],GRB.EQUAL,-peak_power[t+l]-site_prediction[l-1]+power_prediction[l-1])
                                optimal.addConstr(supply_demand,GRB.EQUAL,0)
                            
                            # constraint for finding the peak
                            constP = LinExpr()
                            constP.addTerms(1,PE)
                            constP.addTerms(-1,L[l])
                            optimal.addConstr(constP,GRB.GREATER_EQUAL,0)
                            
                            # constraint for restricting the peak approximately
                            # constraint is not required as it is reducing the chance of increasing the P violation
                            optimal.addConstr(L[l],GRB.LESS_EQUAL,temp_peak[l])
                            
                        # run the optimization code only if at least one of the machine is active
                        if sum(y_active) > 0: 
                            
                            optimal.optimize(); 
                            
                            dd_reduction_optimal = 0
                            # updating the battery storage
                            A -= B.X
                            A += Y.X
                            A = min(A,AM)
                            peak_power[t] -= B.X
                            peak_power[t] -= R.X
                            
                            for j in range(global_file.M):
                                
                                if y[j].X > 0 or v[j].X > 0:
                                    # machine is turned on
                                    machines[j].on_off = 1
                                    if y[j].X > 0:
                                        peak_power[t] += machines[j].on_power
                                        peak_power[t:t+t_dash*2+1] += machines[j].idle_power 
                                     
                                    # it keeps a track of jobs scheduled to be deleted from the queue
                                    jobs_done = []
                                    # if job_scheduled is more than 1, then there is no idle energy consumption
                                    job_scheduled = 0
                                    dd_reduction_flag = [t+l for l in range(t_dash*2+1)]
                                    
                                    time_covered = []
                                    job_scheduled = 0
                                    
                                    for q in range(len(x[j])):
                                        for l in range(len(x[j][q])):
                                            if x[j][q][l].X > 0:
                                                
                                                # schedule jobs if the job is being started at time t
                                                current_part = machines[j].queue[q]
                                                time = current_part.time
                                                power = current_part.power
                                                
                                                flag_peak = 1
                                                
                                                for ll in range(t_dash*2):
                                                    if peak_power[t+ll]+power+site_prediction[ll]-power_prediction[ll] > peak_constraint:
                                                        flag_peak = 1
                                                        break
                                                
                                                if flag_peak == 1:
                                                    jobs_scheduled = 1
                                                    jobs_done.append(q)
                                                    
                                                    peak_power[t+l:t+l+time] += power
                                                    machines[j].scheduled_time = t+l+time
                                                    
                                                    dd_reduction_optimal += len(machines[j].queue)+1-q
                                                
                                                    for tt in range(time):
                                                        time_covered.append(t+l+tt)
                                                    
                                                    # updating the attributes of the part
                                                    current_part = function_file.update_part(current_part,machines[j].scheduled_time)
                                                    if t > 96+48*5:
                                                        operation += 1
                                                    
                                                    # moving the job to the next machine
                                                    if current_part.state < current_part.total_task:
                                                        machines[current_part.machine].queue = function_file.queue_sorting(machines[current_part.machine].queue,current_part,case_priority,t)
                                                    else:
                                                        proportion_completed = throughput/float(0.001+arrival)
                                                        if t > 96+48*5:
                                                            throughput += 1
                                                            dueDate += max(0,machines[j].scheduled_time-current_part.due_date)
                                                    break
                                            
                                    if job_scheduled > 0:
                                        peak_power[time_covered] -= machines[j].idle_power
                                                       
                                    # deleting the jobs that have been scheduled
                                    for jd in sorted(jobs_done, reverse=True): 
                                        del machines[j].queue[jd]      
                                         
                                elif machines[j].on_off == 1 and z[j].X == 1:
                                    machines[j].on_off = 0
                                    peak_power[t] += machines[j].off_power
                                    machines[j].off_time = t+t_dash*2+1
                                    
                            D = max(0,D+dd_reduction-dd_reduction_optimal-delay_lyapunov)
                            P = max(0,P+max(peak_power[t:t+t_dash*2+1])-peak_constraint_lyapunov)
                            #print t,R.X,B.X,A,Y.X,reneP,siteC,peak_power[t],L[0].X,X[0].X,'-----------'#t,P,D,reneP,siteC,sum([L[l].X for l in range(7)])
                            
                        # if no machine is turned on, then just update the battery and D,P
                        else:
                            A += windPower[t/2]*wind_multiply
                            A = min(A,AM)
                            peak_power[t] -= windPower[t/2]*wind_multiply
                            
                            # if not solved and machine is on, then number of machines x 1 = delay added
                            D = max(0,D+sum(y_active)-delay_lyapunov)
                            P = max(0,P+max(peak_power[t:t+t_dash*2+1])-peak_constraint_lyapunov)
                                  
            result[model][reps][0] = arrival
            result[model][reps][1] = throughput
            result[model][reps][2] = operation
            result[model][reps][3] = dueDate/float(throughput+0.01)
            result[model][reps][4] = sum(peak_power[96+48*5:])
            temp = np.partition(-peak_power, 10)
            result_temp = -temp[5:10]
            result[model][reps][5] = np.mean(result_temp)
            cost = np.dot(peak_power,price_nr) 
            result[model][reps][6] = cost/float(1000)+np.mean(result_temp)*22.39

        result_print[model] = np.mean(result[model],0)
        print result_print[model]
        print "model "+str(model)
        
resultDF = pd.DataFrame(result_print,columns=['arrival','throughput','operation','due date','consumption','peak','cost']) 
print case_demand
print resultDF
plt.plot(peak_power[90:1800])
plt.plot(PE_array)
#plt.show()
print np.argmax(peak_power)
if toprint == 1:
    resultDF = pd.DataFrame(result_print,columns=['arrival','throughput','operation','due date','consumption','peak','cost'])
    resultDF.to_excel(resultOutput,'result')
    