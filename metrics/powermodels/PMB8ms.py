from .PM import *
import math

# The power model of an Huawei Technologies Co., Ltd RH2288H V2 (Intel Xeon E502673 v3 8 core, 1 chips, 8 cores/chip)
# The data is accessed from https://www.spec.org/power_ssj2008/results/res2014q2/power_ssj2008-20140408-00655.html

class PMB4ms(PM):
	def __init__(self):
		super().__init__()
		self.powerlist = [68.7, 78.3, 84.0, 88.4, 92.5, 97.3, 104.0, 111.0, 121.0, 131.0, 137.0]

	# cpu consumption in 100
	def power(self):
		cpu = self.host.getCPU()
		index = math.floor(cpu / 10)
		left = self.powerlist[index]
		right = self.powerlist[index + 1 if cpu%10 != 0 else index]
		alpha = (cpu / 10) - index
		return alpha * right + (1 - alpha) * left