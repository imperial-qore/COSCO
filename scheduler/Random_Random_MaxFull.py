from .Scheduler import *
import numpy as np
from copy import deepcopy


class RMScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        return self.RandomContainerSelection()

    def placement(self, containerIDs):
        return self.MaxFullPlacement(containerIDs)
