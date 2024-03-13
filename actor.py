import torch
import numpy as np
from model import DuelingNetwork
from memory import PrioritizedReplayBuffer


class RainbowAgent:
    def __init__(self, state_dim, action_dim, memory_size, batch_size, gamma=0.99, lr=1e-4, target_update=100,
                 epsilon_decay=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learn_step_counter = 0

        self.policy_net = DuelingNetwork(state_dim, action_dim).float()
        self.target_net = DuelingNetwork(state_dim, action_dim).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target_net to evaluation mode

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values, _ = self.policy_net(state)
            action = action_values.argmax().item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)

        # Calculate current Q values
        curr_q, _ = self.policy_net(states)
        curr_q = curr_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Calculate expected Q values
        with torch.no_grad():
            next_q, _ = self.target_net(next_states)
        expected_q = rewards + self.gamma * next_q.max(1)[0] * (1 - dones)

        # Calculate loss and update network
        loss = (curr_q - expected_q.detach()).pow(2) * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # Update target network
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

    def update_priorities(self, batch_indices, batch_priorities):
        self.memory.update_priorities(batch_indices, batch_priorities)
