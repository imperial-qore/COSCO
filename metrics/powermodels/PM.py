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

	def power(self):
		return 0

	def temperature(self):
		p = self.power()
		r, c = 0.5, 0.03 
		t_ambient = 20
		# thermal resistance and capacitance are typical values 
		# taken from 'Wolf, M., 2016. The physics of computing. Elsevier.'
		# temperature is calculated using CRAC model
		temp = p * r + t_ambient + t_ambient * math.exp(r * c)
		return temp