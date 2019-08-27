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
aloha_rewards = {}
tdma_rewards = {}
agent_throughputs = {}
aloha_throughputs = {}
tdma_throughputs = {}
max_iter = 100000
N = 2000
q = 0.4
t = 0.4
R = 4
n_runs = 8

x = np.linspace(0, max_iter, max_iter)
for i in range(0, n_runs):
	agent_rewards[i]     = np.loadtxt('rewards/M40/agent_len1e5_M40_h2_q0.4_%d.txt' % (i+1))
	aloha_rewards[i]     = np.loadtxt('rewards/M40/aloha_len1e5_M40_h2_q0.4_%d.txt' % (i+1))
	tdma_rewards[i]     = np.loadtxt('rewards/M40/tdma_len1e5_M40_h2_q0.4_%d.txt' % (i+1))
	agent_throughputs[i] = cal_throughput(max_iter, N, agent_rewards[i])
	aloha_throughputs[i] = cal_throughput(max_iter, N, aloha_rewards[i])
	tdma_throughputs[i]  = cal_throughput(max_iter, N, tdma_rewards[i])



my_agent_throughputs = np.array(agent_throughputs[0])
my_aloha_throughputs = np.array(aloha_throughputs[0])
my_tdma_throughputs  = np.array(tdma_throughputs[0])
for j in range(1, n_runs):
	my_agent_throughputs = np.vstack((my_agent_throughputs, agent_throughputs[i]))
	my_aloha_throughputs = np.vstack((my_aloha_throughputs, aloha_throughputs[i]))
	my_tdma_throughputs  = np.vstack((my_tdma_throughputs, tdma_throughputs[i]))



total_throughputs = my_agent_throughputs + my_aloha_throughputs + my_tdma_throughputs
total_mean = np.mean(total_throughputs, axis=0)
total_std = np.std(total_throughputs, axis=0)

agent_mean = np.mean(my_agent_throughputs, axis=0)
agent_std = np.std(my_agent_throughputs, axis=0)
aloha_mean = np.mean(my_aloha_throughputs, axis=0)
aloha_std = np.std(my_aloha_throughputs, axis=0)
tdma_mean = np.mean(my_tdma_throughputs, axis=0)
tdma_std = np.std(my_tdma_throughputs, axis=0)




if q > 1/(R+1):
	total_optimal = np.ones(max_iter) * (q + (1-q)*(R-1)/R) * (1-t) + t * (1-q)
	agent_optimal = np.ones(max_iter) * (1-q)*(R-1)/R * (1-t)
	aloha_optimal = np.ones(max_iter) * q * (1-t)
	tdma_optimal  = np.ones(max_iter) * t * (1-q)
else:
	total_optimal = np.ones(max_iter) * (1 - q)
	agent_optimal = np.ones(max_iter) * (1 - q) * (1-t)
	aloha_optimal = np.zeros(max_iter)
	tdma_optimal  = np.ones(max_iter) * t * (1-q)


### plot
# fig = plt.figure(figsize=(10, 7))
fig = plt.figure()
ax  = fig.add_subplot(111)



# agent_line, = plt.plot(agent_mean, color='b', lw=1.5, label='DRL')

# aloha_line, = plt.plot(aloha_mean, color='g', lw=1.5, label='q-ALOHA')
# # agent_optimal_line, = plt.plot(agent_optimal, color='b', lw=1, linestyle='-.', label='agent optimal')
# tdma_line, = plt.plot(tdma_mean, color='r', lw=1.5, label='TDMA')
# tdma_optimal_line, = plt.plot(tdma_optimal, color='r', lw=1, linestyle='-.', label='agent optimal')
total_line, = plt.plot(total_mean, color='k', lw=1.5, label='Sum')
total_optimal_line, = plt.plot(total_optimal, color='k', lw=1.5, linestyle='-.', label='Optimal sum')

plt.fill_between(x, total_mean-total_std, total_mean+total_std,    
    alpha=0.2, edgecolor='k', facecolor='k',
    linewidth=1, linestyle='--', antialiased=True)

# plt.fill_between(x, agent_mean-agent_std, agent_mean+agent_std,    
#     alpha=0.2, edgecolor='b', facecolor='b',
#     linewidth=1, linestyle='--', antialiased=True)

# plt.fill_between(x, aloha_mean-aloha_std, aloha_mean+aloha_std,    
#     alpha=0.2, edgecolor='g', facecolor='g',
#     linewidth=1, linestyle='--', antialiased=True)

# plt.fill_between(x, tdma_mean-tdma_std, tdma_mean+tdma_std,    
#     alpha=0.2, edgecolor='r', facecolor='r',
#     linewidth=1, linestyle='--', antialiased=True)


handles, labels = ax.get_legend_handles_labels()

plt.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5,1), fancybox=True, shadow=True)
# plt.fill_between(x, agent_mean-agent_std, agent_mean+agent_std,    
#     alpha=0.5, edgecolor='#e50000', facecolor='#e50000',
#     linewidth=4, linestyle='dashdot', antialiased=True)
# plt.fill_between(x, tdma_mean-tdma_std, tdma_mean+agent_std,    
#     alpha=0.5, edgecolor='#0343df', facecolor='#0343df',
#     linewidth=4, linestyle='dashdot', antialiased=True)
# plt.fill_between(x, tdma_mean-tdma_std, tdma_mean+tdma_std,    
#     alpha=0.5, edgecolor='#15b01a', facecolor='#15b01a',
#     linewidth=4, linestyle='dashdot', antialiased=True)






plt.xlabel('Time steps')
plt.ylabel('Throughput')
plt.xlim((0, max_iter))
plt.ylim((0, 0.9))
# plt.savefig('figs/DRL+qALOHA+TDMA_20180906_2.png')
# plt.savefig('figs/DRL+TDMA_20180905.eps')
plt.show()


