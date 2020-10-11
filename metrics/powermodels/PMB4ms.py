from .PM import *
import math

# The power model of an IBM Corporation IBM System x3350 (Intel Xeon E502673 v3 4 core, 1 chips, 4 cores/chip)
# The data is accessed from https://www.spec.org/power_ssj2008/results/res2008q2/power_ssj2008-20080506-00052.html

class PMB4ms(PM):
	def __init__(self):
		super().__init__()
		self.powerlist = [71.0, 77.9, 83.4, 89.2, 95.6, 102.0, 108.0, 114.0, 119.0, 123.0, 126.0]

	# cpu consumption in 100
	def power(self):
		cpu = self.host.getCPU()
		index = math.floor(cpu / 10)
		left = self.powerlist[index]
		right = self.powerlist[index + 1 if cpu%10 != 0 else index]
		alpha = (cpu / 10) - index
		return alpha * right + (1 - alpha) * left