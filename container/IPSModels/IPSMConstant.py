from .IPSM import *

class IPSMConstant(IPSM):
	def __init__(self, constant_ips):
		super().__init__()
		self.constant_ips = constant_ips

	def ips(self):
		return self.constant_ips if self.container.env.interval <= 10 else 0