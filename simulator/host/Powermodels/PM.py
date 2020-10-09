import math

class PM():
	def __init__(self):
		self.host = None

	def allocHost(self, h):
		self.host = h

	# cpu consumption in 100
	def powerFromCPU(self, cpu):
		index = math.floor(cpu / 10)
		left = self.powerlist[index]
		right = self.powerlist[index + 1 if cpu%10 != 0 else index]
		alpha = (cpu / 10) - index
		return alpha * right + (1 - alpha) * left