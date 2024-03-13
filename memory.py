import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, prior_eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.filled = False
        self.prior_eps = prior_eps

    def add(self, state, action, reward, next_state, done, error):
        """Store a new transition in the buffer and update the priorities list.

        Args:
            state: The state observed from the environment.
            action: The action taken in response to the state.
            reward: The reward received after taking the action.
            next_state: The next state observed after taking the action.
            done: A boolean indicating whether the episode has ended.
            error: The priority or importance of this transition.
        """
        max_priority = max(self.priorities) if self.buffer else 1.0  # Use the max priority for new transitions

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority if error is None else error  # Set the priority of the new transition

        self.pos = (self.pos + 1) % self.capacity  # Update the position for the next addition

    def sample(self, batch_size, beta=0.4):
        if self.filled:
            probs = self.priorities
        else:
            probs = self.priorities[:self.pos]

        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(
            dones), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (priority + self.prior_eps) ** self.alpha

    def __len__(self):
        return len(self.buffer) if self.filled else self.pos

    def sample_batch(self, beta, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("The replay buffer is empty!")

        # Calculate priorities powered by alpha for sampling
        priorities_pow_alpha = np.power(self.priorities[:len(self.buffer)], self.alpha)
        sample_probs = priorities_pow_alpha / priorities_pow_alpha.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), size=batch_size, p=sample_probs)

        # Extract sampled transitions
        transitions = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = np.power(total * sample_probs[indices], -beta)
        weights /= weights.max()  # Normalize weights

        # Unzip the transitions to separate states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*transitions)

        return transitions, indices, weights

