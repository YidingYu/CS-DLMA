import numpy as np
import matplotlib.pyplot as plt






def my_plot(file):
	action = np.loadtxt(file)

	max_iter = 100000
	N = 100000
	x = np.linspace(0, max_iter, max_iter)
	# plt.scatter(x[-N:], action[-N:], c='b', marker='s', alpha=0.4)
	plt.scatter(x[:N], action[:N], c='b', marker='s', alpha=0.4)
	# plt.ylim(-0.5, 4.5)


for i in range(1, 3):
	plt.figure(figsize=(36, 5))
	my_plot('rewards/action_len1e5_M20_q0.4_t2-5_alpha1_g0.95_e0.001_r10_%d.txt' % i)
plt.show()