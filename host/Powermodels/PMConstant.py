from .PM import *

class PMConstant(PM):
	def __init__(self, constant):
		super().__init__()
		self.constant = constant

	# CPU consumption in 100
	def power(self):
		return self.constant