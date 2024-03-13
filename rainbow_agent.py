import torch
import torch.nn as nn
import torch.nn.functional as F
from transitions import Transition
import torch.optim as optim
import numpy as np
from memory import PrioritizedReplayBuffer  # Ensure this import is correct
from network import Network  # Ensure you have this import if Network is in a separate file

class RainbowAgent:
    def __init__(self, state_dim, action_dim, memory_size, batch_size, target_update, gamma=0.99,
                 lr=1e-4, epsilon_decay=1e-3, alpha=0.6, beta=0.4, v_min=0.0, v_max=200.0, atom_size=51, seed=42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.beta = beta
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.seed = seed
        self.learn_step_counter= 50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = PrioritizedReplayBuffer(memory_size, alpha)  # Ensure alpha is used correctly in your memory

        self.dqn = Network(state_dim, action_dim, atom_size, torch.linspace(v_min, v_max, atom_size)).to(self.device)
        self.dqn_target = Network(state_dim, action_dim, atom_size, torch.linspace(v_min, v_max, atom_size)).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)


    def choose_action(self, state):
        if state is None:
            raise ValueError("Received None state in choose_action. State must not be None.")
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.dqn(state_tensor)
            action = action_values.argmax(1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        default_priority = 1.0
        self.memory.add(state, action, reward, next_state, done, default_priority)
        # self.memory.add(state, action, reward, next_state, done)  # Assuming 'add' is the correct method name


    def sample_memory(self):
        # Inside RainbowAgent's sample_memory method or wherever you call sample_batch
        transitions, indices, weights = self.memory.sample_batch(self.beta, self.batch_size)

        #transitions, indices, weights = self.memory.sample_batch(self.beta)
        batch = Transition(*zip(*transitions))
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(np.array(batch.action)).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).view(-1, 1).to(self.device)
        weights = torch.FloatTensor(weights).view(-1, 1).to(self.device)
        return states, actions, rewards, next_states, dones, indices, weights

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones, indices, weights = self.sample_memory()

        current_q_distributions = self.dqn.dist(states)
        actions = actions.long().unsqueeze(-1).expand(-1, -1, self.atom_size)
        current_q_distributions = current_q_distributions.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.dqn(next_states).max(1)[0]  # This should already give you a 1D tensor of max Q-values for each state in the batch
            best_actions = self.dqn(next_states).argmax(1)  # Ensure this matches the expected shape
            next_q_distribution = self.dqn_target.dist(next_states)
            #print(next_q_values.shape)  # Add this line
            #best_actions = next_q_values.argmax(1)
            all_next_q_values = self.dqn(next_states)  # Obtain Q-values for all actions
            best_actions = all_next_q_values.argmax(1)

            #best_actions = next_q_values.argmax(1)
            best_actions = best_actions.view(-1, 1, 1).expand(-1, -1, self.atom_size)
            next_q_distribution = next_q_distribution.gather(1, best_actions).squeeze(1)

            rewards = rewards.expand_as(next_q_distribution)
            dones = dones.expand_as(next_q_distribution)
            supports = self.dqn.support.expand_as(next_q_distribution)
            Tz = rewards + (1 - dones) * self.gamma * supports
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / ((self.v_max - self.v_min) / (self.atom_size - 1))
            l = b.floor().long()
            u = b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atom_size - 1)) * (l == u)] += 1

            m = states.new_zeros(self.batch_size, self.atom_size)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atom_size), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atom_size).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_q_distribution * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_q_distribution * (b - l.float())).view(-1))

        loss = -torch.sum(m * current_q_distributions.log(), 1)
        loss = (loss * weights).mean()

        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()

        #new_priorities = loss.detach().cpu().numpy() + self.memory.prior_eps
        new_priorities = np.full(len(indices), loss.item() + self.memory.prior_eps)
        self.memory.update_priorities(indices, new_priorities)

    def update_target_network(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def choose_action(self, state):
        if state is None:
            raise ValueError("Received None state in choose_action method.")

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.dqn.eval()  # Set the network to evaluation mode
        with torch.no_grad():
            # Convert state to tensor, add batch dimension, send to device, etc.
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Use the network to choose the action with the highest Q-value
            action = self.dqn(state_tensor).argmax(1).item()  # Assuming discrete action space
        self.dqn.train()  # Set back to training mode
        return action

    def reset(self):
        # Initialize your game or environment state
        self.state = self.initial_state()  # This should set up the initial state correctly
        return self.state  # Make sure this is not None

    def save_model(self, file_path):
        """Save the current model state.

        Args:
            file_path (str): The path to the file where the model state will be saved.
        """
        torch.save(self.dqn.state_dict(), file_path)
