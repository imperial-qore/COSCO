from .DM import *

class DMConstant(DM):
	def __init__(self, constant_size, constant_read, constant_write):
		super().__init__()
		self.constant_size = constant_size
		self.constant_read = constant_read
		self.constant_write = constant_write

	def diskIO(self):
		return self.constant_size, self.constant_read, self.constant_write