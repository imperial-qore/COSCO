from .PM import *

class PMConstant(PM):
	def __init__(self, constant):
		super().__init__()
		self.constant = constant
		self.powerlist = [constant] * 11

	# CPU consumption in 100
	def power(self):
		return self.constant