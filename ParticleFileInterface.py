import struct


class ParticleFileInterface:
	def __init__(self):
		self.data = [] # data[i][j][k]: i-th frame, j-th particle, k-th dim
	
	def Clear(self):
		self.data.clear()

	def PushFrame(self, in_poses, pn):
		# Note: deep copy
		poses = []
		for pi in range(pn):
			pos = [
				in_poses[pi][0],
				in_poses[pi][1],
				in_poses[pi][2]
			]
			poses.append(pos)
		self.data.append(poses)

	def Export(self, filename):
		# Open File
		f = open(filename, mode="wb")

		# Pack data
		bytes = b""
		## frame number
		bytes += struct.pack("I", len(self.data))

		## particle positions per frame
		for fi in range(len(self.data)):
			## particle number
			bytes += struct.pack("I", len(self.data[fi]))

			## particle positions
			for pi in range(len(self.data[fi])):
				## x, y, z
				bytes += struct.pack("fff", 
					self.data[fi][pi][0],
					self.data[fi][pi][1],
					self.data[fi][pi][2],
					)

		# Write data
		f.write(bytes)

		# Close File
		f.close()