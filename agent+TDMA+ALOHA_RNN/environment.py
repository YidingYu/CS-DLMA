import numpy as np 
import time



class ENVIRONMENT(object):
	"""docstring for ENVIRONMENT"""
	def __init__(self,
				 features = 10,
				 aloha_ratio=4,
				 tdma_ratio=4,
				 n_actions = 5,
				 transmission_prob = 0.4,
				 penalty=0.5
				 ):
		super(ENVIRONMENT, self).__init__()
		self.features = features
		self.n_nodes = 3
		self.n_actions = n_actions


		self.aloha_ratio = aloha_ratio
		self.tdma_ratio = tdma_ratio

		self.action_list = [0, 1, 0, 0, 1] 
		# self.action_list = [0, 0, 0, 0, 1] 
		self.counter = 0
		self.tdma_action = self.action_list[self.counter]
		self.tdma_counter = 0
		self.tdma_success_counter = 0
		self.tdma_duration = aloha_ratio
		
		self.attempt_prob = transmission_prob
		self.aloha_action = 0
		self.aloha_counter = 0
		self.aloha_success_counter = 0
		self.aloha_duration = tdma_ratio
		self.agent_counter = 0
		self.agent_success_counter = 0

		self.penalty = penalty
		

	def reset(self):
		init_state = np.zeros(self.features, int)
		return init_state

	def step(self, action, duration):

		reward = 0
		agent_reward = 0
		aloha_reward = 0
		tdma_reward = 0
		# print(self.tdma_action)

		if action == 0:
			if self.aloha_action == 0 and self.tdma_action == 0:
				observation_ = 'I'
			elif self.aloha_action == 1 and self.tdma_action == 0:
				self.aloha_success_counter += 1
				observation_ = 'B'
			elif self.aloha_action == 0 and self.tdma_action == 1:
				self.tdma_success_counter += 1
				observation_ = 'B'
			else:
				observation_ = 'B'
		elif action > 1:
			self.agent_counter += 1
			if self.aloha_action == 0 and self.tdma_action == 0:
				self.agent_success_counter += 1
			observation_ = 'U'

		else: # action == 1	
			self.agent_counter += 1
			if self.aloha_action == 0 and self.tdma_action == 0:
				self.agent_success_counter += 1
			if self.agent_counter == duration:
				if self.agent_success_counter == duration:
					reward = duration - self.penalty
					agent_reward = duration - self.penalty
					observation_ = 'S'
				else:
					observation_ = 'C'
				self.agent_counter = 0
				self.agent_success_counter = 0
			else:
				# never come to here
				observation_ = 'C'				

		# prepare for aloha
		self.aloha_counter += 1
		if self.aloha_counter == self.aloha_duration:
			if self.aloha_success_counter == self.aloha_duration:
				reward = self.aloha_duration - self.penalty
				aloha_reward = self.aloha_duration - self.penalty
			if np.random.random()<self.attempt_prob:
				self.aloha_action = 1
			else:
				self.aloha_action = 0			
			self.aloha_counter = 0
			self.aloha_success_counter = 0

		# prepare for tdma
		self.tdma_counter += 1
		if self.tdma_counter == self.tdma_duration:
			if self.tdma_success_counter == self.tdma_duration:
				reward = self.tdma_duration - self.penalty
				tdma_reward = self.tdma_duration - self.penalty
			self.tdma_counter = 0
			self.tdma_success_counter = 0
			self.counter += 1
			if self.counter == len(self.action_list):
				self.counter = 0
			self.tdma_action = self.action_list[self.counter]	
		return observation_, reward, agent_reward, aloha_reward, tdma_reward

# test
# env = ENVIRONMENT()
# # agent_action = [0, 0, 1, 0,  0, 1, 0, 0,  0, 0, 0, 0]
# for i in range(50):
# 	if i % 4 == 0:
# 		action = 4
# 	elif i % 4 == 1:
# 		action = 3
# 	elif i % 4 == 2:
# 		action = 2
# 	elif i % 4 == 3:
# 		action = 1

# 	env.step(action, 4, i)


