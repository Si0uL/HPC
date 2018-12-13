import matplotlib.pyplot as plt
from math import sqrt

with open('epsilon3.txt', 'r') as _file:
	lines = _file.readlines()

for n, l in enumerate(lines):
	lines[n] = map(float, l.strip().split(','))

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

# ----------------------------------------------------------------------

with open('size3.txt', 'r') as _file:
	lines = _file.readlines()

for n, l in enumerate(lines):
	lines[n] = map(float, l.strip().split(','))

sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
threads = [1, 2, 4, 8, 16, 32, 64]

Times_epsilon = [
	[line[-1] for line in lines if line[2] == thread]
	for thread in threads
]

plt.figure()
for idx, curve in enumerate(Times_epsilon):
	plt.plot(sizes, map(sqrt, curve), "-o", label="{} Threads".format(threads[idx]))

plt.legend()
plt.xlabel("Size of matrix")
plt.ylabel("Square Root of Execution Time")
plt.show()
