
class Scheduler():
	def __init__(self):
		self.env = None

	def setEnvironment(self, env):
		self.env = env

	def selection(self):
		pass

	def placement(self, containerlist):
		pass

	def getMigrationFromHost(hostID, decision):
		containerIDs = []
		for (cid, _) in decision:
			hid = self.getContainerByID(cid).getHostID()
			if hid == hostID:
				containerIDs.append(cid)
		return containerIDs

	def getMigrationToHost(hostID, decision):
		containerIDs = []
		for (cid, hid) in decision:
			if hid == hostID:
				containerIDs.append(cid)
		return containerIDs