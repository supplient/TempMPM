import time

class MySimpleTimer:
	def __init__(self):
		self.prevTime = time.time()

	def tick(self)->float:
		curTime = time.time()
		diff = curTime - self.prevTime
		self.prevTime = curTime
		return diff