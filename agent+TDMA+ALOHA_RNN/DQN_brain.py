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
from keras.layers import Dense, Input, LSTM
from keras.initializers import he_normal


class DQN:
	def __init__(self, 
				features=5,
				aloha_ratio=4,
				tdma_ratio=4,
				n_actions=2,
				n_nodes=3,
				
				history_len=20,
				memory_size=1000,
				replace_target_iter=200,
				batch_size=32,
				learning_rate=0.01,
				gamma=0.9,
				epsilon=0.1,
				epsilon_min=0.01,
				epsilon_decay=0.995,
				alpha=0
				):
		
		
		self.features = features
		self.aloha_ratio = aloha_ratio
		self.tdma_ratio = tdma_ratio
		self.n_actions = n_actions
		self.n_nodes = n_nodes
		
		self.history_len = history_len
		self.memory_size = memory_size
		self.replace_target_iter = replace_target_iter
		self.batch_size = batch_size
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.learn_step_counter = 0
		self.alpha = alpha
		self.memory = deque(maxlen=self.memory_size)

		self.model = self.build_model()
		self.target_model = self.build_model()
		self.loss = []



	def build_model(self):
		inputs = Input(shape=(None, self.features, ))
		h1 = LSTM(64, activation='relu', return_sequences=False, kernel_initializer=he_normal(seed=247))(inputs)
		h2 = Dense(64, activation='relu', kernel_initializer=he_normal(seed=447))(h1)
		outputs = Dense(self.n_actions*self.n_nodes, kernel_initializer=he_normal(seed=3447))(h2) 
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(loss='mse', optimizer='rmsprop')
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
		state = state.reshape(1, -1, self.features)
		self.epsilon *= self.epsilon_decay
		self.epsilon  = max(self.epsilon_min, self.epsilon)
		if np.random.random() < self.epsilon:
			return np.random.randint(0, self.n_actions)
		action_values = self.model.predict(state)
		return self.alpha_function(action_values[0])


	def add_experience(self, experience):
		self.memory.append(experience)


	def repalce_target_parameters(self):
		weights = self.model.get_weights()
		self.target_model.set_weights(weights)


	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.repalce_target_parameters() # iterative target model
		self.learn_step_counter += 1

		if len(self.memory) < self.memory_size and len(self.memory) > self.history_len:
			sample_index = np.random.choice(len(self.memory)-(self.history_len), size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_size-(self.history_len), size=self.batch_size)      

		batch_memory = []
		for i in range(self.batch_size):
			batch_memory.append(list(self.memory)[sample_index[i]:(sample_index[i]+self.history_len)])
		batch_memory = np.array(batch_memory)

		state    = batch_memory[:, :self.history_len, :self.features]
		action   = batch_memory[:, -1, self.features].astype(int)
		duration = batch_memory[:, -1, self.features+1]
		reward1  = batch_memory[:, -1, self.features+2]
		reward2  = batch_memory[:, -1, self.features+3]
		reward3  = batch_memory[:, -1, self.features+4]
		next_state = batch_memory[:, -self.history_len:, -self.features:]
		batch_index = np.arange(self.batch_size, dtype=np.int32)

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
		plt.savefig('figs/{}.png'.format(i))
