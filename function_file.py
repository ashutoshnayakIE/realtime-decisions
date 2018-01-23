import math
import global_file
import numpy as np

# appending the jobs in the queue based on their priority
# priority is assigned based on the requirement of the manager

def queue_sorting(queue,part,case_priority,t):
    '''
    print [queue[x].start for x in range(len(queue))],'-------',case_priority, part.due_date
    print [queue[x].due_date for x in range(len(queue))],'-------',part.machine,part.start
    print [queue[x].time for x in range(len(queue))],'-------',part.min_start_time
    print [queue[x].due_date/float(queue[x].remaining_time) for x in range(len(queue))],'-------'
    '''
    if len(queue) == 0:
        queue.append(part)
    else:
        for q in range(len(queue)):
            # FCFS
            if case_priority == 1:
                if part.start < queue[q].start:
                    queue.insert(q,part)
                    break
                elif q == len(queue) - 1:
                    queue.append(part)
            # EDD
            elif case_priority == 2:
                if part.due_date < queue[q].due_date:
                    queue.insert(q,part)
                    break
                elif q == len(queue) - 1:
                    queue.append(part)
            # SPT
            elif case_priority == 3:
                if part.time < queue[q].time:
                    queue.insert(q,part)
                    break
                elif q == len(queue) - 1:
                    queue.append(part)
            # CR
            if case_priority == 4:
                if (part.due_date-t)/float(part.remaining_time) < (queue[q].due_date-t)/float(queue[q].remaining_time):
                    queue.insert(q,part)
                    break
                elif q == len(queue) - 1:
                    queue.append(part)

    return queue

def update_part(part,time):
    part.state += 1
    part.min_start_time = time
    if part.state < part.total_task:
        part.machine = global_file.process[part.part_type][1+part.state][0]
        part.time = global_file.process[part.part_type][1+part.state][1]
        part.power = global_file.process[part.part_type][1+part.state][2]
        part.remaining_time -= global_file.process[part.part_type][1+part.state-1][1]
    
    return part
        
def energy(t,t_dash,wind_multiply,windPower):
    w1 = []
    s1 = []
    windP = windPower[t-27:t+1]
    p1 = 1.6506
    p2 = -0.7574
    p3 = 0.1438
    p4 = -0.0337
    p24 = 0.1541
    
    for tt in range(t_dash):
        v = 0
        v += p1*windP[-1]
        v -= p1*p24*windP[-25]
        v += p2*windP[-2]
        v -= p2*p24*windP[-26]
        v += p3*windP[-3]
        v -= p3*p24*windP[-27]
        v += p4*windP[-24]
        v -= p4*p24*windP[-28]
        v += p24*windP[-24]
        w1.append(v)
        windP.append(v)
        
        s1.append(global_file.site[(global_file.t+tt)%24])
        
    n_trials = 10
    trial_valuesW = np.zeros([n_trials,t_dash],dtype='float')
    trial_valuesS = np.zeros([n_trials,t_dash],dtype='float')
    
    for w2 in range(len(w1)):
        sigma = math.sqrt(0.03)
        trial_valuesW[:,w2] = np.transpose(np.random.normal(w1[w2]*wind_multiply,sigma*1.4,n_trials))
        trial_valuesS[:,w2] = np.transpose(np.random.normal(s1[w2],2,n_trials))
        
    w3 = sum(trial_valuesW)/n_trials 
    s3 = sum(trial_valuesS)/n_trials
    w4 = []
    s4 = []
    for t in range(t_dash):
        w4.append(w3[t])
        w4.append(w3[t])
        s4.append(s3[t])
        s4.append(s3[t]) 
    return w4,s4
