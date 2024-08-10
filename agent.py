import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Agent:
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.00025, 
                 gamma=0.9, 
                 epsilon=1.0, 
                 eps_decay=0.99999975, 
                 eps_min=0.1, 
                 replay_buffer_capacity=100_000, 
                 batch_size=32, 
                 sync_network_rate=10000):
        
        self.num_actions = num_actions
        self.learn_step_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Networks
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss() # Feel free to try this loss function instead!

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        #=========================================
        # Implement the epsilon greedy algortihm 
        # pick a random number from 0-1, if it is less than
        # epsilon, select a random action and return it

            # YOUR CODE HERE

        #=========================================
        # Passing in a list of numpy arrays is slower than creating a tensor from a numpy array
        # Hence the `np.array(observation)` instead of `observation`
        # observation is a LIST of numpy arrays because of the LazyFrame wrapper
        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
         #Return the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))
        
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        #======================================================
        # Check to see if the replay buffer is enough. If it is 
        # less than the batch_size, return 

        #======================================================
        
        self.sync_networks()
        
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        #=======================================================
        # Predict Q-Values for the actions taken in the current states

        # Pass the current states through the online network to get the predicted Q-values for all actions.
        # You should obtain a tensor of shape (batch_size, n_actions) representing the Q-values for each action in each state.
        predicted_q_values = None # CHANGE THIS

        # From the predicted Q-values, select the Q-value corresponding to the action that was actually taken.
        # Hint: Use np.arange to create an array of indices for the batch, and actions.squeeze() to get the action indices.
        # The result should be an array of Q-values for the actions that were actually taken.
        predicted_q_values = None # CHANGE THIS

        # Calculate Target Q-Values for the next states

        # Use the target network to predict the Q-values for all actions for the next states.
        # Then, select the maximum Q-value for each next state. This represents the best expected future reward.
        target_q_values = None #CHANGE THIS

        # Calculate the target Q-values using the formula: reward + (gamma * max_future_q_value * (1 - done))
        # Hint: Use the .max() function. Note that returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        # This accounts for immediate rewards and discounted future rewards, adjusting for terminal states with 'done'.
        # If 'done' is True (indicating the episode has ended), future rewards are set to 0 by multiplying with (1 - done).
        target_q_values = None #CHANGE THIS
        #=======================================================
        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()


        


