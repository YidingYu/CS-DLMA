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



def main(history_len, n_actions, ratio, max_iter):
	start = time.time()

	agent_reward_list = []
	wifi_reward_list  = []
	reward_list = []
	action_list = []

	### save the number of visits to state
	# state_list = []
	# num_current_state = []
	# num_state_list = []
	# num_state = 0

	channel_state = env.reset()
	channel_state_length = len(channel_state)	
	
	# initialize state
	state = deque(maxlen=history_len) # mutliple (history_len) channel states
	for i in range(history_len):
		state.append(channel_state) # initialize state (input of RNN)
	
	# temporary experience array for reward backpropagation (not experience buffer in DQN)
	experience_array = np.zeros((ratio, channel_state_length*2+4))

	observation = 'I' # initialize the channel to be idle
	
	for slot in tqdm(range(max_iter)):
		if observation =='I': 
			action = dqn_agent.choose_action(np.array(state)) 
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
			action_list.append(-1)

		observation, reward, agent_reward, wifi_reward = env.step(action, duration)

		reward_list.append(reward)
		agent_reward_list.append(agent_reward)
		wifi_reward_list.append(wifi_reward)
		
		# store new experience
		if observation != 'U':
			next_channel_state = np.hstack([return_action(origin_action, n_actions), return_observation(observation)])
			experience_array = np.vstack((experience_array[1:], 
										np.concatenate([channel_state, 
														[origin_action, duration, agent_reward, wifi_reward], 
														next_channel_state])))
			if origin_action == 0 and wifi_reward > 1: # for reward backpropagation (see WCNC paper)
				experience_array[-math.ceil(reward):, channel_state_length+3] = wifi_reward / math.ceil(reward)
			dqn_agent.add_experience(experience_array[0])
			if slot > 200:
				dqn_agent.learn()       
			channel_state = next_channel_state 
			state.append(channel_state)

			### save the number of visits to state
			# num_current_state.append(state_list.count(str(np.array(state))))
			# if state_list.count(str(np.array(state))) == 0:
			# 	num_state += 1
			# num_state_list.append(num_state)
			# state_list.append(str(np.array(state)))

	with open('rewards/agent_len1e5_M20_W2_alpha50_g0.999_MM6_r10_1.txt', 'w') as my_agent:
		for i in agent_reward_list:
			my_agent.write(str(i) + '   ')
	with open('rewards/wifi_len1e5_M20_W2_alpha50_g0.999_MM6_r10_1.txt', 'w') as my_wifi:
		for i in wifi_reward_list:
			my_wifi.write(str(i) + '   ') 
	with open('rewards/action_len1e5_M20_W2_alpha50_g0.999_MM6_r10_1.txt', 'w') as my_action:
		for i in action_list:
			my_action.write(str(i) + '   ') 
	with open('rewards/reward_len1e5_M20_W2_alpha50_g0.999_MM6_r10_1.txt', 'w') as my_reward:
		for i in reward_list:
			my_reward.write(str(i) + '   ') 

	#### save the number of visits to state
	# with open('states/current_state_len1e5_M20_W2_alpha50_g0.999_MM6_r10_1.txt', 'w') as my_current_state:
	# 	for i in num_current_state:
	# 		my_current_state.write(str(i) + '	')
	# with open('states/total_state_len1e5_M20_W2_alpha50_g0.999_MM6_r10_1.txt', 'w') as my_total_state:
	# 	for i in num_state_list:
	# 		my_total_state.write(str(i) + '	')


	print('-----------------------------')
	print('average agent reward:{}'.format(np.mean(agent_reward_list[-2000:])))
	print('average wifi reward: {}'.format(np.mean(wifi_reward_list[-2000:])))
	print('average total reward:{}'.format(np.mean(agent_reward_list[-2000:]) + 
											np.mean(wifi_reward_list[-2000:])))
	print('Time elapsed:', time.time()-start)

	### save training loss
	# dqn_agent.my_plot('len1e5_M20_W2_alpha50_g0.999_MM6_r10_1')


if __name__ == "__main__":
	RATIO = 10 # the packet length of WiFi
	NUM_ACTIONS = 11 # the number of actions 0-10
	env = ENVIRONMENT(features=NUM_ACTIONS+4, 
					  ratio = RATIO,
					  n_actions = NUM_ACTIONS,
					  init_wifi_window_size=2,
					  max_backoff = 6,
					  penalty = 0.5
					  )

	dqn_agent = DQN(env.features,
					env.ratio,
					env.n_actions,
					env.n_nodes, 
					history_len=20,
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
	main(dqn_agent.history_len, env.n_actions, RATIO, max_iter=100000)


