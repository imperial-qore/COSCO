from .RM import *

class RMConstant(RM):
	def __init__(self, constant_size, constant_read, constant_write):
		self.constant_size = constant_size
		self.constant_read = constant_read
		self.constant_write = constant_write

	def ram(self):
		return self.constant_size, self.constant_read, self.constant_write