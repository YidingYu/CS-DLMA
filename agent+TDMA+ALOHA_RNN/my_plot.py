import numpy as np
import matplotlib.pyplot as plt


### calculate throughput
def cal_throughput(max_iter, N, reward):
	temp_sum = 0
	throughput = np.zeros(max_iter)
	for i in range(max_iter):
		if i < N:
			temp_sum     += reward[i] 
			throughput[i] = temp_sum / (i+1)
		else:
			temp_sum  += reward[i] - reward[i-N]
			throughput[i] = temp_sum / N
	return throughput


agent_rewards = {}
csma_rewards = {}
agent_throughputs = {}
csma_throughputs = {}
max_iter = 50000
N = 1000
x = np.linspace(0, max_iter, max_iter)

q = 0.4
R = 4

total_optimal = np.ones(max_iter) * (q + (1-q)*(R-1)/R)
agent_optimal = np.ones(max_iter) * (1-q)*(R-1)/R
csma_optimal = np.ones(max_iter) * q

for i in range(0, 4):
	agent_rewards[i]     = np.loadtxt('rewards/agent_len5e4_q0.5_M20_h4_e1-0.005_r4_a5_p0.5_%d.txt' % (i+1))
	csma_rewards[i]     = np.loadtxt('rewards/csma_len5e4_q0.5_M20_h4_e1-0.005_r4_a5_p0.5_%d.txt' % (i+1))
	agent_throughputs[i] = cal_throughput(max_iter, N, agent_rewards[i])
	csma_throughputs[i] = cal_throughput(max_iter, N, csma_rewards[i])




my_agent_throughputs = np.array([agent_throughputs[0], agent_throughputs[1], agent_throughputs[2], agent_throughputs[3]])
my_csma_throughputs  = np.array([csma_throughputs[0], csma_throughputs[1], csma_throughputs[2], csma_throughputs[3]])

total_throughputs = my_agent_throughputs + my_csma_throughputs
total_mean = np.mean(total_throughputs, axis=0)
total_std = np.std(total_throughputs, axis=0)

agent_mean = np.mean(my_agent_throughputs, axis=0)
agent_std = np.std(my_agent_throughputs, axis=0)
csma_mean = np.mean(my_csma_throughputs, axis=0)
csma_std = np.std(my_csma_throughputs, axis=0)



### plot
# fig = plt.figure(figsize=(10, 7))
fig = plt.figure()
ax  = fig.add_subplot(111)



agent_line, = plt.plot(agent_mean, color='b', lw=1.5, label='DRL')
agent_optimal_line, = plt.plot(agent_optimal, color='b', lw=1, linestyle='-.', label='Optimal DRL')
csma_line, = plt.plot(csma_mean, color='r', lw=1.5, label='q-csma (q0.4)')
csma_optimal_line, = plt.plot(csma_optimal, color='r', lw=1, linestyle='-.', label='Optimal q-csma')
total_line, = plt.plot(total_mean, color='k', lw=1.5, label='Sum')
total_optimal_line, = plt.plot(total_optimal, color='k', lw=1.5, linestyle='-.', label='Optimal sum')

plt.fill_between(x, total_mean-total_std, total_mean+total_std,    
    alpha=0.2, edgecolor='k', facecolor='k',
    linewidth=1, linestyle='--', antialiased=True)
plt.fill_between(x, agent_mean-agent_std, agent_mean+agent_std,    
    alpha=0.2, edgecolor='b', facecolor='b',
    linewidth=1, linestyle='--', antialiased=True)
plt.fill_between(x, csma_mean-csma_std, csma_mean+csma_std,    
    alpha=0.2, edgecolor='r', facecolor='r',
    linewidth=1, linestyle='--', antialiased=True)


handles, labels = ax.get_legend_handles_labels()

plt.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5,1), fancybox=True, shadow=True)
# plt.fill_between(x, agent_mean-agent_std, agent_mean+agent_std,    
#     alpha=0.5, edgecolor='#e50000', facecolor='#e50000',
#     linewidth=4, linestyle='dashdot', antialiased=True)
# plt.fill_between(x, csma_mean-csma_std, csma_mean+agent_std,    
#     alpha=0.5, edgecolor='#0343df', facecolor='#0343df',
#     linewidth=4, linestyle='dashdot', antialiased=True)
# plt.fill_between(x, csma_mean-csma_std, csma_mean+csma_std,    
#     alpha=0.5, edgecolor='#15b01a', facecolor='#15b01a',
#     linewidth=4, linestyle='dashdot', antialiased=True)






plt.xlabel('Time steps')
plt.ylabel('Throughput')
plt.xlim((0, max_iter))
plt.ylim((0, 1.3))
plt.savefig('figs/DRL+q-csma_q0.6_20180907.png')
plt.savefig('figs/DRL+q-csma_q0.6_20180907.eps')
plt.show()


