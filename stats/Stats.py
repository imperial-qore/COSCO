
class Stats():
	def __init__(self, Environment, WorkloadModel, Datacenter, Scheduler):
		self.env = Environment
		self.workload = WorkloadModel
		self.datacenter = Datacenter
		self.scheduler = Scheduler
		self.initStats()

	def initStats(self):	
		self.hostinfo = []
		self.containerinfo = []
		self.schedulerinfo = []

	def saveHostInfo(self, interval):
		

	def saveStats(self, interval):	
