from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
from rl_tutorial.utils.Buffer import ReplayBuffer

class DDQN_Agent(object):
    def __init__(self, env, cfg,
                 mem_size=1000000):
        
        # Initialize the env variables
        n_actions = env.action_space.n
        input_dims = env.observation_space.shape[0]
        self.nstates = input_dims
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions

         # Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)

        self.gamma = float(data['gamma']) if float(data['gamma']) else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if float(data['epsilon']) else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if float(data['epsilon_min']) else 0.05
        self.epsilon_dec = float(data['epsilon_decay']) if float(data['epsilon_decay']) else 0.995
        self.learning_rate = float(data['learning_rate']) if float(data['learning_rate']) else 0.001
        self.batch_size = int(data['batch_size']) if int(data['batch_size']) else 32
        self.model_file = data['model_file'] if data['model_file'] else 'DDQN_Agent_saved'
        self.replace_target = int(data['replace_target']) if int(data['replace_target']) else 100
        
        # Build the memory buffer
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        
        # Build the evaluation and target models
        self.q_net = self.build_dqn(self.learning_rate, n_actions, input_dims, 256, 256)
        self.q_target_net = self.build_dqn(self.learning_rate, n_actions, input_dims, 256, 256)


    def build_dqn(self, lr, n_actions, input_dims, nodes1, nodes2):
        model = Sequential([
                    Dense(nodes1, input_shape=(input_dims,)),
                    Activation('relu'),
                    Dense(nodes2),
                    Activation('relu'),
                    Dense(n_actions)])

        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def action(self, state):
        state = np.reshape(state, (1, self.nstates))
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_net.predict_on_batch(state)
            action = np.argmax(actions)

        return action, 0
    
    def play(self, state):
        state = np.reshape(state, (1, self.nstates))
        actions = self.q_net.predict_on_batch(state)
        action = np.argmax(actions)
        return action, 0

    def train(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            # Action indices store indices for one-hot encoding of the actions that were taken 
            action_indices = np.dot(action, action_values)
            
            # Get Q(s') using target model
            q_next = self.q_target_net.predict_on_batch(new_state)
            # Get Q(s') using evaluation model
            q_eval = self.q_net.predict_on_batch(new_state)
            # Get Q(s)
            q_pred = self.q_net.predict_on_batch(state)
            
            # Actions that will be taken at next state s'
            max_actions = np.argmax(q_net, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            
            # target_q = r + gamma * q'  
            # only change the indices [one-hot] that correspond to current action
            # only take q' for the action indice that will be taken at the next state
            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done
            
            # The loss function for this training with be mse between the predicted q by the evaluation model and the target Q calculated above
            _ = self.q_net.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target_net.set_weights(self.q_net.get_weights())

    def save_model(self):
        self.q_net.save(self.model_file)

    def load_model(self):
        self.q_net = load_model(self.model_file)
        if self.epsilon == 0.0:
            self.update_network_parameters()