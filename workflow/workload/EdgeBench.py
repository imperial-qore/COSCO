from .Workload import *
from datetime import datetime
from framework.database.Database import *
from random import gauss, choices
import random

class EdgeBench(Workload):
    def __init__(self, num_workloads, std_dev, database):
        super().__init__()
        self.num_workloads = num_workloads
        self.std_dev = std_dev
        self.db = database
        self.workflows = ['Type1', 'Type2', 'Type3']
        self.applications = ['shreshthtuli/yolo', 'shreshthtuli/pocketsphinx', 'shreshthtuli/aeneas']
        self.appdict = dict(zip(['Y', 'P', 'A'], self.applications))

    def generateNewWorkflows(self, interval):
        workflowlist = []
        minimum_workflows = 1 if interval == 0 else 0
        for i in range(max(minimum_workflows,int(gauss(self.mean, self.sigma)))):
            WorkflowID = self.workflow_id
            workflow = choices(self.workflows, weights=[0.5, 0.4, 0.4])[0]
            SLA = np.random.randint(10,15)
            workflowlist.append((WorkflowID, interval, SLA, workflow))
            self.workflow_id += 1
        return workflowlist

    def generateContainer(self, WorkflowID, dependentOn, interval, sla, application, app):
        CreationID = self.creation_id
        self.creation_id += 1
        return CreationID, (WorkflowID, dependentOn, interval, sla, application, app)

    def generateNewContainers(self, interval):
        workloadlist = []
        workflowlist = self.generateNewWorkflows(interval)
        for i, (WorkflowID, interval, SLA, workflow) in enumerate(workflowlist):
            if workflow == 'Type1': 
                # P -> A -> A -> P
                CreationID1, info = self.generateRandomContainer(WorkflowID, None, interval, SLA, workflow, self.appdict['P'])
                workloadlist.append((CreationID1, *info))
                CreationID2, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, SLA, workflow, self.appdict['A'])
                workloadlist.append((CreationID2, *info))
                CreationID3, info = self.generateRandomContainer(WorkflowID, [CreationID2], interval, SLA, workflow, self.appdict['A'])
                workloadlist.append((CreationID3, *info))
                CreationID4, info = self.generateRandomContainer(WorkflowID, [CreationID3], interval, SLA, workflow, self.appdict['P'])
                workloadlist.append((CreationID4, *info))
            if workflow == 'Type2': 
                #      P
                # A ->   -> P
                #      Y 
                CreationID1, info = self.generateRandomContainer(WorkflowID, None, interval, SLA, workflow, self.appdict['A'])
                workloadlist.append((CreationID1, *info))
                CreationID2, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, SLA, workflow, self.appdict['P'])
                workloadlist.append((CreationID2, *info))
                CreationID3, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, SLA, workflow, self.appdict['Y'])
                workloadlist.append((CreationID3, *info))
                CreationID4, info = self.generateRandomContainer(WorkflowID, [CreationID2, CreationID3], interval, SLA, workflow, self.appdict['P'])
                workloadlist.append((CreationID4, *info))
            if workflow == 'Type3': 
                #      A -> Y
                # A ->   
                #      P -> P
                CreationID1, info = self.generateRandomContainer(WorkflowID, None, interval, SLA, workflow, self.appdict['A'])
                workloadlist.append((CreationID1, *info))
                CreationID2, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, SLA, workflow, self.appdict['A'])
                workloadlist.append((CreationID2, *info))
                CreationID3, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, SLA, workflow, self.appdict['P'])
                workloadlist.append((CreationID3, *info))
                CreationID4, info = self.generateRandomContainer(WorkflowID, [CreationID2], interval, SLA, workflow, self.appdict['Y'])
                workloadlist.append((CreationID4, *info))
                CreationID5, info = self.generateRandomContainer(WorkflowID, [CreationID3], interval, SLA, workflow, self.appdict['P'])
                workloadlist.append((CreationID5, *info))
        self.createdContainers += workloadlist
        self.deployedContainers += [False] * len(workloadlist)
        return self.getUndeployedContainers()