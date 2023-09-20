from .IPSM import *

class IPSMBitbrain(IPSM):
    def __init__(self, ips_list, max_ips, duration, SLA):
        super().__init__()
        self.ips_list = ips_list
        self.max_ips = max_ips
        self.SLA = SLA
        self.duration = duration
        self.completedInstructions = 0
        self.completedAfterMigration = 0
        self.totalInstructions = 0

    def getIPS(self):
        if self.totalInstructions == 0:
            for ips in self.ips_list[:self.duration]: self.totalInstructions += ips * self.container.env.intervaltime
        if self.completedInstructions < self.totalInstructions:
            return self.ips_list[(self.container.env.interval - self.container.startAt) % len(self.ips_list)]
        return 0
    
    def getMaxIPS(self):
        return self.max_ips