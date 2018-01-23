import global_file
import numpy as np
class part:
    def __init__(self, part_type,start, state,min_start_time,due_date):
        process = global_file.process
        self.part_type = part_type
        self.start = start
        self.state = state
        self.machine = process[part_type][1+state][0]
        self.time = process[part_type][1+state][1]
        self.power = process[part_type][1+state][2]
        self.total_task = process[part_type][0][0]
        self.remaining_time = process[part_type][0][1]
        self.due_date = due_date
        self.min_start_time = min_start_time
        self.last_day = 0

class machine:
    def __init__(self, number, queue, on_off, on_time, on_power, off_power, idle_power, scheduled_time, off_time):
        self.number = number
        self.queue = queue
        self.on_off = on_off                    # 0/1 if on or off
        self.on_time = on_time                  # time when it may be on
        self.on_power = on_power                # power consumed when on
        self.off_power = off_power              # power consumed when off
        self.idle_power = idle_power            # power consumed when idle
        self.scheduled_time = scheduled_time    # time till which the jobs have been scheduled
        self.off_time = off_time