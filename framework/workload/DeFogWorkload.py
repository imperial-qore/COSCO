from .Workload import *
from datetime import datetime
from framework.database.Database import *
from random import gauss, choices
import random

class DFW(Workload):
    def __init__(self, num_workloads, std_dev, database):
        super().__init__()
        self.num_workloads = num_workloads
        self.std_dev = std_dev
        self.db = database
        
    def generateNewContainers(self, interval):
        workloadlist = []
        containers = []
        applications = ['shreshthtuli/yolo', 'shreshthtuli/pocketsphinx', 'shreshthtuli/aeneas']
        for i in range(max(1,int(gauss(self.num_workloads, self.std_dev)))):
            CreationID = self.creation_id
            SLA = np.random.randint(5,8) ## Update this based on intervals taken
            application = random.choices(applications, weights=[0.2, 0.4, 0.4])[0]
            workloadlist.append((CreationID, interval, SLA, application))
            self.creation_id += 1
        self.createdContainers += workloadlist
        self.deployedContainers += [False] * len(workloadlist)
        return self.getUndeployedContainers()