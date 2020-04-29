from .Scheduler import *
import numpy as np
from copy import deepcopy


class MaRScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        selectedIDs = self.MaxUseSel()
        return selectedIDs

    def placement(self, containerIDs):
        decision = []
        for cid in containerIDs:
            decision.append((cid, np.random.randint(0, len(self.env.hostlist))))
        return decision
