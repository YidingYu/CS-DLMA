import os
import numpy as np
from collections import deque
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2 # percentage of using the GPU
set_session(tf.Session(config=config))

from keras.models import Model
from keras.layers import Dense, Input, Add
from keras.initializers import he_normal



class DQN:
	def __init__(self, 
				state_size,
				aloha_ratio=4,
				tdma_ratio=4,
				n_actions=5,
				n_nodes=3,
				memory_size=500,
				replace_target_iter=200,
				batch_size=32,
				learning_rate=0.01,
				gamma=0.9,
				epsilon=1,
				epsilon_min=0.01,
				epsilon_decay=0.995,
				alpha=0
				):
		# hyper-parameters
		self.state_size = state_size
		self.n_actions = n_actions
		self.aloha_ratio = aloha_ratio
		self.tdma_ratio = tdma_ratio
		self.n_nodes = n_nodes
		self.memory_size = memory_size
		self.replace_target_iter = replace_target_iter
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay       
		self.alpha = alpha
		self.memory = np.zeros((self.memory_size, self.state_size * 2 + 5)) # memory_size * len(s, a, d r, s_)
		# temporary parameters
		self.learn_step_counter = 0
		self.memory_couter = 0
				
		# # # # # # # build mode
		self.model        = self.build_ResNet_model() # model: evaluate Q value
		self.target_model = self.build_ResNet_model() # target_mode: target network
		self.loss = []
		

	def build_ResNet_model(self):
		inputs = Input(shape=(self.state_size, ))
		# h1 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=256547), kernel_regularizer=regularizers.l2(0.01))(inputs) #h1
		# h2 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=257), kernel_regularizer=regularizers.l2(0.01))(h1) #h2

		h1 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=247))(inputs) #h1
		h2 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=447))(h1) #h2

		h3 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=403))(h2) #h3
		h4 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=43))(h3) #h4
		add1 = Add()([h4, h2])
		
		h5 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4035))(add1) #h5
		h6 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=42203))(h5) #h6
		add2 = Add()([h6, add1])

		h7 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=455403))(add2) #h5
		h8 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=40123))(h7) #h6
		add3 = Add()([h8, add2])

		h9 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=405433))(add3) #h3
		h10 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4103))(h9) #h4
		add4 = Add()([h10, add3])
		
		h11 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=12403))(add4) #h5
		h12 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=40352))(h11) #h6
		add5 = Add()([h12, add4])

		h13 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=402223))(add5) #h5
		h14 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4043))(h13) #h6
		add6 = Add()([h14, add5])

		h15 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=40873))(add6) #h3
		h16 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4903))(h15) #h4
		add7 = Add()([h16, add6])
		
		h17 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=7403))(add7) #h5
		h18 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=45603))(h17) #h6
		add8 = Add()([h18, add7])

		h19 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=40593))(add8) #h5
		h20 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=405633))(h19) #h6
		add9 = Add()([h20, add8])

		# h21 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=404863))(add9) #h3
		# h22 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4043))(h21) #h4
		# add10 = Add()([h22, add9])
		
		# h23 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=456403))(add10) #h5
		# h24 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=73))(h23) #h6
		# add11 = Add()([h24, add10])

		# h25 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4031))(add11) #h5
		# h26 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4203))(h25) #h6
		# add12 = Add()([h26, add11])

		# h27 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=40312))(add12) #h3
		# h28 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=40943))(h27) #h4
		# add13 = Add()([h28, add12])
		
		# h29 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=40763))(add13) #h5
		# h30 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=1403))(h29) #h6
		# add14 = Add()([h30, add13])

		# h31 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4203))(add14) #h5
		# h32 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=4403))(h31) #h6
		# add15 = Add()([h32, add14])

		# h33 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=5403))(add15) #h3
		# h34 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=33403))(h33) #h4
		# add16 = Add()([h34, add15])
		
		# h35 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=6403))(add16) #h5
		# h36 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=403))(h35) #h6
		# add17 = Add()([h36, add16])

		# h37 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=362403))(add17) #h5
		# h38 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=403263))(h37) #h6
		# add18 = Add()([h38, add17])

		# h39 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=3403))(add18) #h5
		# h40 = Dense(64, activation="relu", kernel_initializer=he_normal(seed=76403))(h39) #h6
		# add19 = Add()([h40, add18])
		
		outputs =  Dense(self.n_actions*self.n_nodes, kernel_initializer=he_normal(seed=347))(add9)
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(loss="mse", optimizer='rmsprop')
		return model

	def alpha_function(self, action_values):
		action_values_list = []
		for j in range(self.n_actions):
			if action_values[3*j] < 0:
				action_values[3*j] = 0.0001
			if action_values[3*j+1] < 0:
				action_values[3*j+1] = 0.0001				
			if action_values[3*j+2] < 0:
				action_values[3*j+2] = 0.0001
		if self.alpha == 1:
			for j in range(self.n_actions):
				action_values_list.append(np.log(action_values[3*j])+np.log(action_values[3*j+1])+np.log(action_values[3*j+2]))
		elif self.alpha == 0:
			for j in range(self.n_actions):
				action_values_list.append(action_values[3*j]+action_values[3*j+1]+action_values[3*j+2])
		elif self.alpha == 100: # alpha = infinity
			for j in range(self.n_actions):
				action_values_list.append(min(action_values[3*j], action_values[3*j+1], action_values[3*j+2]))	
		else:
			for j in range(self.n_actions):
				action_values_list.append(1/(1-self.alpha) * (action_values[3*j]**(1-self.alpha) + action_values[3*j+1]**(1-self.alpha) + action_values[3*j+2]**(1-self.alpha)))
		return np.argmax(action_values_list)

	def choose_action(self, state):
		state = state[np.newaxis, :]
		self.epsilon *= self.epsilon_decay
		self.epsilon  = max(self.epsilon_min, self.epsilon)
		if np.random.random() < self.epsilon:
			return np.random.randint(0, self.n_actions)
		action_values = self.model.predict(state)
		return self.alpha_function(action_values[0])


	def store_transition(self, transition): # s_: next_state
		if not hasattr(self, 'memory_couter'):
			self.memory_couter = 0
		index = self.memory_couter % self.memory_size
		self.memory[index, :] = transition
		self.memory_couter   += 1


	def repalce_target_parameters(self):
		weights = self.model.get_weights()
		self.target_model.set_weights(weights)


	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.repalce_target_parameters() # iterative target model
		self.learn_step_counter += 1

		if self.memory_couter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_couter, size=self.batch_size)        
		batch_memory = self.memory[sample_index, :]

		state      = batch_memory[:, :self.state_size]
		action     = batch_memory[:, self.state_size].astype(int) # float -> int
		duration   = batch_memory[:, self.state_size+1].astype(int)
		reward1     = batch_memory[:, self.state_size+2]
		reward2     = batch_memory[:, self.state_size+3]
		reward3     = batch_memory[:, self.state_size+4]
		next_state = batch_memory[:, -self.state_size:]

		batch_index = np.arange(self.batch_size, dtype=np.int64)


		# DQN loss: (r + max_a q_targ(s_{t+1}, a) - q(s_t, a_t))^2
		q = self.model.predict(state) # state		
		q_targ = self.target_model.predict(next_state) # next state

		for i in range(len(action)):
			action_ = self.alpha_function(q_targ[i])
			q[i][3*action[i]]   = reward1[i] / duration[i] * (1-self.gamma**duration[i]) / (1-self.gamma) + self.gamma**duration[i] * q_targ[i][3*action_]
			q[i][3*action[i]+1] = reward2[i] / duration[i] * (1-self.gamma**duration[i]) / (1-self.gamma) + self.gamma**duration[i] * q_targ[i][3*action_+1]
			q[i][3*action[i]+2] = reward3[i] / duration[i] * (1-self.gamma**duration[i]) / (1-self.gamma) + self.gamma**duration[i] * q_targ[i][3*action_+2]


		history = self.model.fit(state, q, self.batch_size, epochs=1, verbose=0)
		self.loss.append(history.history['loss'])


	def my_plot(self, i):
		import matplotlib.pyplot as plt
		plt.plot(self.loss)
		plt.savefig('figs/result_{}.png'.format(i))
		# plt.show()