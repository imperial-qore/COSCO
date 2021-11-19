from .Workload import *
from datetime import datetime
from framework.database.Database import *
from random import gauss, choices
import random

class AIoTW(Workload):
    def __init__(self, num_workloads, std_dev, database):
        super().__init__()
        self.num_workloads = num_workloads
        self.std_dev = std_dev
        self.db = database
        
    def generateNewContainers(self, interval):
        workloadlist = []
        containers = []
        applications = ['resnet18', 'resnet34', 'squeezenet1_0', 'mobilenet_v2', 'mnasnet1_0', 'googlenet', 'resnext50_32x4d']
        multiplier = np.array([2, 1, 4, 2, 1, 3, 1])
        weights = 1 - (multiplier / np.sum(multiplier))
        for i in range(max(1,int(gauss(self.num_workloads, self.std_dev)))):
            CreationID = self.creation_id
            SLA = np.random.randint(5,8) ## Update this based on intervals taken
            application = random.choices(applications, weights=weights)[0]
            workloadlist.append((CreationID, interval, SLA, application))
            self.creation_id += 1
        self.createdContainers += workloadlist
        self.deployedContainers += [False] * len(workloadlist)
        return self.getUndeployedContainers()