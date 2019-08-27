import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches





def my_plot(file1, file2):
	current_state = np.loadtxt(file1)
	total_state = np.loadtxt(file2)
	plt.plot(current_state, color='b',  label='Current')
	plt.plot(total_state, color='r', label='Total')


for i in range(1, 2):
	plt.figure(figsize=(12, 5))
	my_plot('states/current_state_len1e5_M20_g0.99_q0.4_t2-5_alpha0_r10_%d.txt' % i, 
			'states/total_state_len1e5_M20_g0.99_q0.4_t2-5_alpha0_r10_%d.txt' % i)
plt.show()