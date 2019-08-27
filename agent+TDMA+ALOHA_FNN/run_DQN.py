from environment import ENVIRONMENT
from DQN_brain import DQN
from collections import deque
import numpy as np
from tqdm import tqdm

import math
import time


def return_observation(observation):
	if observation == 'B':
		return [0, 0, 0, 1]
	elif observation == 'I':
		return [0, 0, 1, 0]
	elif observation == 'S':
		return [0, 1, 0, 0]
	elif observation == 'C':
		return [1, 0, 0, 0] 

def return_action(action, NUM_ACTIONS):
	one_hot_vector = [0 for _ in range(NUM_ACTIONS)]
	one_hot_vector[action] = 1
	return one_hot_vector


def main(n_actions, ratio, max_iter):
	start = time.time()

	agent_reward_list = []
	aloha_reward_list = []
	tdma_reward_list  = []
	reward_list = []
	action_list = []
	state = env.reset()
	state_length = len(state)

	### save the number of visits to state
	# state_list = []
	# num_current_state = []
	# num_state_list = []
	# num_state = 0

	experience_array = np.ones((ratio, state_length*2+5)) # +2: action, duration, reward
	experience_array[:, state_length+1] = 1
	print('------------------------------------------')
	print('---------- Start processing ... ----------')
	print('------------------------------------------')
	observation = 'I'
	for i in tqdm(range(max_iter)):
		if observation == 'I': 
			action = dqn_agent.choose_action(state)
			duration = action if action > 0 else 1
			origin_action = action
			action_list.append(action)
		elif observation == 'U':
			action -= 1
			action_list.append(origin_action)
		else: 
			action = 0
			duration = 1
			origin_action = 0
			action_list.append(action)

		observation, reward, agent_reward, aloha_reward, tdma_reward, \
		= env.step(action, duration)

		reward_list.append(reward)
		agent_reward_list.append(agent_reward)
		aloha_reward_list.append(aloha_reward)
		tdma_reward_list.append(tdma_reward)

 
		# store new experience
		if observation != 'U':        
			if state_length<n_actions+5:
				next_state = np.hstack([return_action(origin_action, n_actions), return_observation(observation)])
			else:
				next_state = np.concatenate([state[n_actions+4:], return_action(origin_action, n_actions), return_observation(observation)])
			experience_array = np.vstack((experience_array[1:], np.concatenate([state, [origin_action, duration, agent_reward, aloha_reward, tdma_reward], next_state])))

			if origin_action == 0 and aloha_reward > 1:
				experience_array[-math.ceil(reward):, state_length+3] = aloha_reward / math.ceil(reward)
			if origin_action == 0 and tdma_reward > 1:
				experience_array[-math.ceil(reward):, state_length+4] = tdma_reward / math.ceil(reward)

			# dqn_agent.store_transition(experience_array[0, :state_length], experience_array[0, state_length], experience_array[0, state_length+1], experience_array[0, state_length+2], experience_array[0, -state_length:])
			dqn_agent.store_transition(experience_array[0])
			if i > 200:
				dqn_agent.learn()
			state = next_state

			### save the number of visits to state
			# num_current_state.append(state_list.count(str(np.array(state))))
			# if state_list.count(str(np.array(state))) == 0:
			# 	num_state += 1
			# num_state_list.append(num_state)
			# state_list.append(str(np.array(state)))

	with open('rewards/agent_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1.txt', 'w') as my_agent:
		for i in agent_reward_list:
			my_agent.write(str(i) + '   ')
	with open('rewards/aloha_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1.txt', 'w') as my_aloha:
		for i in aloha_reward_list:
			my_aloha.write(str(i) + '   ') 
	with open('rewards/tdma_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1.txt', 'w') as my_tdma:
		for i in tdma_reward_list:
			my_tdma.write(str(i) + '   ') 
	with open('rewards/action_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1.txt', 'w') as my_action:
		for i in action_list:
			my_action.write(str(i) + '   ') 
	with open('rewards/reward_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1.txt', 'w') as my_reward:
		for i in reward_list:
			my_reward.write(str(i) + '   ') 

	### save the number of visits to state
	# with open('states/current_state_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1.txt', 'w') as my_current_state:
	# 	for i in num_current_state:
	# 		my_current_state.write(str(i) + '	')
	# with open('states/total_state_len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1.txt', 'w') as my_total_state:
	# 	for i in num_state_list:
	# 		my_total_state.write(str(i) + '	')



	print('-----------------------------')
	print('average agent reward:{}'.format(np.mean(agent_reward_list[-2000:])))
	print('average aloha reward:{}'.format(np.mean(aloha_reward_list[-2000:])))
	print('average tdma  reward:{}'.format(np.mean(tdma_reward_list[-2000:])))
	print('average total reward:{}'.format(np.mean(agent_reward_list[-2000:]) + 
											np.mean(aloha_reward_list[-2000:]) +
											np.mean(tdma_reward_list[-2000:])))
	print('Time elapsed:', time.time()-start)
	dqn_agent.my_plot('len1e5_M20_g0.999_q0.5_t2-5_alpha50_r10_h20_1')



if __name__ == "__main__":
	RATIO1 = RATIO # ALOHA packet length
	RATIO2 = RATIO # TDMA packet length
	NUM_ACTIONS = RATIO1 + 1
	env = ENVIRONMENT(state_size=300,  # 15*20
					  aloha_ratio = RATIO1,
					  tdma_ratio = RATIO2,
					  n_actions = NUM_ACTIONS,
					  transmission_prob=0.5,
					  penalty=0.5,
					  )

	dqn_agent = DQN(env.state_size,
					env.aloha_ratio,
					env.tdma_ratio,
					env.n_actions,  
					env.n_nodes,
					memory_size=1000,
					replace_target_iter=20,
					batch_size=32,
					learning_rate=0.01,
					gamma=0.999,
					epsilon=1,
					epsilon_min=0.005,
					epsilon_decay=0.995,
					alpha=50
					)

	main(env.n_actions, RATIO1, max_iter=100000)