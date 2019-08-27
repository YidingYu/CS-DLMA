import numpy as np
import matplotlib.pyplot as plt


def my_plot(file1, file2, file3, R, X, Y, q, H):
	max_iter = 100000
	N = 100000

	# load reward
	agent1_reward = np.loadtxt(file1)
	agent2_reward = np.loadtxt(file2)
	agent3_reward = np.loadtxt(file3)

	throughput_agent1 = np.zeros((1, max_iter))
	throughput_agent2 = np.zeros((1, max_iter))
	throughput_agent3 = np.zeros((1, max_iter))


	if q > 1/(R-H+1):
		total_optimal = np.ones(max_iter) * ((1-q)*(R-H-1)/R * (1-X/Y) + q * (1-X/Y) * (R-H)/R + X/Y * (1-q) * (R-H)/R)
		agent_optimal = np.ones(max_iter) * (1-q)*(R-H-1)/R * (1-X/Y)
		aloha_optimal = np.ones(max_iter) * q * (1-X/Y) * (R-H)/R
		tdma_optimal  = np.ones(max_iter) * X/Y * (1-q) * (R-H)/R
		# print('total', total_optimal[0])
		# print('agent', agent_optimal[0])
		# print('aloha', aloha_optimal[0])
		# print('tdma', tdma_optimal[0])
	else:
		total_optimal = np.ones(max_iter) * ((1-X/Y) * (1-q) * (R-H-1)/R + X/Y * (1-q) * (R-H)/R)
		agent_optimal = np.ones(max_iter) * (1-X/Y) * (1-q) * (R-H-1)/R
		aloha_optimal = np.zeros(max_iter)
		tdma_optimal  = np.ones(max_iter) * X/Y * (1-q) * (R-H)/R

	agent1_temp_sum = 0
	agent2_temp_sum = 0
	agent3_temp_sum = 0
	for i in range(0, max_iter):
		if i < N:
			agent1_temp_sum += agent1_reward[i]
			agent2_temp_sum += agent2_reward[i]
			agent3_temp_sum += agent3_reward[i]
			throughput_agent1[0][i] = agent1_temp_sum / (i+1)
			throughput_agent2[0][i] = agent2_temp_sum / (i+1)
			throughput_agent3[0][i] = agent3_temp_sum / (i+1)
		else:
			agent1_temp_sum += agent1_reward[i] - agent1_reward[i-N]
			agent2_temp_sum += agent2_reward[i] - agent2_reward[i-N]
			agent3_temp_sum += agent3_reward[i] - agent3_reward[i-N]
			throughput_agent1[0][i] = agent1_temp_sum / N
			throughput_agent2[0][i] = agent2_temp_sum / N
			throughput_agent3[0][i] = agent3_temp_sum / N

	plt.xlim((0, max_iter))
	plt.ylim((-0.05, 0.4))

	agent1_line, = plt.plot(throughput_agent1[0], color='r', lw=1.2, label='agent')
	agent2_line, = plt.plot(throughput_agent2[0], color='b', lw=1.2, label='aloha')
	agent3_line, = plt.plot(throughput_agent3[0], color='g', lw=1.2, label='tdma')
	# total_line,  = plt.plot(throughput_agent1[0]+throughput_agent2[0]+throughput_agent3[0], color='k', lw=1.2, label='total')

	agent5_line, = plt.plot(agent_optimal, color='r', lw=1, label='agent optimal')
	agent6_line, = plt.plot(aloha_optimal, color='b', lw=1, label='aloha optimal')
	agent7_line, = plt.plot(tdma_optimal, color='g', lw=1, label='tdma optimal')
	# agent8_line, = plt.plot(total_optimal, color='k', lw=1, label='total optimal')
	# plt.grid()
	# plt.legend(handles=[agent1_line, agent2_line, total_line], loc='best')
	# print('---------------')
	# print('agent', np.mean(throughput_agent1[0][-100:]))
	# print('aloha', np.mean(throughput_agent2[0][-100:]))




for i in range(1,3):
	plt.figure(i)
	my_plot('rewards/agent_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h40_%d.txt' % i,
		    'rewards/aloha_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h40_%d.txt' % i,
		     'rewards/tdma_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h40_%d.txt' % i, R=10, X=2, Y=5, q=0.5, H=0.5)
plt.show()