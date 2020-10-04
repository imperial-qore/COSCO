
class RAM():
	# Size = Size of RAM in MB
	# Read = Read speed in MBps
	# Write = Write speed in MBps
	def __init__(self, Size, Read, Write):
		self.size = Size
		self.read = Read
		self.write = Write