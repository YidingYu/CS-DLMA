import numpy as np
import matplotlib.pyplot as plt






def my_plot(file):
	action = np.loadtxt(file)

	max_iter = 20000
	N = 300
	x = np.linspace(0, max_iter, max_iter)
	plt.scatter(x[-N:], action[-N:], c='b', marker='s', alpha=0.4)
	plt.ylim(-0.5, 4.5)


for i in range(3, 5):
	plt.figure(figsize=(36, 5))
	my_plot('rewards/agent_len2e4_env2_q0.4_M20_h2_e1_0.001_0.9995_r4_a2_p0_%d.txt' % i)

plt.show()