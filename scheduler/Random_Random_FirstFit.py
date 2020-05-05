from .Scheduler import *
import numpy as np
from copy import deepcopy


class RFScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        return self.RandomContainerSelection()

    def placement(self, containerIDs):
        return self.FirstFitPlacement(containerIDs)
