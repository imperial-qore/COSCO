from .RM import *

class RMConstant(RM):
	def __init__(self, constant_size, constant_read, constant_write):
		self.size = constant_size
		self.read = constant_read
		self.write = constant_write

	def ram(self):
		return self.size, self.read, self.write