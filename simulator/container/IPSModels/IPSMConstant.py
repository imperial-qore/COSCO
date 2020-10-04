from .IPSM import *

class IPSMConstant(IPSM):
	def __init__(self, constant_ips, max_ips, duration, SLA):
		super().__init__()
		self.constant_ips = constant_ips
		self.max_ips = max_ips
		self.SLA = SLA
		self.duration = duration
		self.completedInstructions = 0

	def getIPS(self):
		self.totalInstructions = self.constant_ips * self.duration * self.container.env.intervaltime
		if self.completedInstructions < self.totalInstructions:
			return self.constant_ips
		return 0

	def getMaxIPS(self):
		return self.max_ips