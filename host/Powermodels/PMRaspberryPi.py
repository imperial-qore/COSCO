from .PM import *
import math

# Power consumption of Raspberry Pi 3 Model B
# @source Kaup, Fabian, Philip Gottschling, and David Hausheer.
# "PowerPi: Measuring and modeling the power consumption of the Raspberry Pi."
# In 39th Annual IEEE Conference on Local Computer Networks, pp. 236-243. IEEE, 2014.

class PMRaspberryPi(PM):
	def __init__(self):
		super().__init__()
		self.powerlist = [0.45, 0.78, 0.90, 1.03, 1.65, 1.66, 1.69, 1.71, 1.72, 1.74, 1.75]

	# cpu consumption in 100
	def power(self):
		cpu = self.host.getCPU()
		index = math.floor(cpu / 10)
		left = self.powerlist[index]
		right = self.powerlist[index + 1]
		alpha = (cpu / 10) - index
		return alpha * right + (1 - alpha) * left