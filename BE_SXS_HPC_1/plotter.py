import matplotlib.pyplot as plt
from math import sqrt

plt.rcParams.update({'font.size': 20})
plt.close('all')

with open('epsilon.txt', 'r') as _file:
	lines = _file.readlines()

for n, l in enumerate(lines):
	lines[n] = list(map(float, l.strip().split(',')))

epsilons = [5, 6, 7, 8, 9, 10, 11]
threads = [1, 2, 4, 8, 16, 32, 64]

Times_epsilon = [
	[line[-1] for line in lines if line[2] == thread]
	for thread in threads
]

plt.figure()
for idx, curve in enumerate(Times_epsilon):
	plt.plot(epsilons, curve, "-o", label="{} Threads".format(threads[idx]))

plt.legend()
plt.xlabel("-log(epsilon)")
plt.ylabel("Execution Time")
plt.show()

Acceleration_epsilon = [
	[Times_epsilon[0][idx] / elt for idx, elt in enumerate(series)]
	for series in Times_epsilon
]

plt.figure()
for idx, curve in enumerate(Acceleration_epsilon):
	plt.plot(epsilons, curve, "-o", label="{} Threads".format(threads[idx]))

plt.legend()
plt.xlabel("-log(epsilon)")
plt.ylabel("Acceleration")
plt.show()

# ----------------------------------------------------------------------

with open('size.txt', 'r') as _file:
	lines = _file.readlines()

for n, l in enumerate(lines):
	lines[n] = list(map(float, l.strip().split(',')))

sizes = [256, 512, 1024, 2048, 4096, 8192]
threads = [1, 2, 4, 8, 16, 32]

Times_sizes = [
	[line[-1] for line in lines if line[2] == thread]
	for thread in threads
]

plt.figure()
for idx, curve in enumerate(Times_sizes):
	plt.plot(sizes, curve, "-o", label="{} Threads".format(
		threads[idx]))

plt.legend()
plt.xlabel("Size of matrix")
plt.ylabel("Execution Time")
plt.show()

Acceleration_size = [
	[Times_sizes[0][idx] / elt for idx, elt in enumerate(series)]
	for series in Times_sizes
]

plt.figure()
for idx, curve in enumerate(Acceleration_size):
	plt.plot(sizes, curve, "-o", label="{} Threads".format(threads[idx]))

plt.legend()
plt.xlabel("Size of matrix")
plt.ylabel("Acceleration")
plt.show()
