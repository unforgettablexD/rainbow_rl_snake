import pygame
from snake_game_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE  # Make sure this is correctly imported
from rainbow_agent import RainbowAgent  # Verify this matches your file and class name
import numpy as np
import torch
import network as network

# Initialize Pygame
pygame.init()


def get_state(game):
    head = game.head
    food = game.food

    # Directions
    dir_left = game.direction == Direction.LEFT
    dir_right = game.direction == Direction.RIGHT
    dir_up = game.direction == Direction.UP
    dir_down = game.direction == Direction.DOWN

    # Points around the snake's head to check for danger
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    # Check for immediate danger
    danger_left = game.is_collision(pt=point_l)
    danger_right = game.is_collision(pt=point_r)
    danger_up = game.is_collision(pt=point_u)
    danger_down = game.is_collision(pt=point_d)

    # Food direction
    food_left = food.x < head.x
    food_right = food.x > head.x
    food_up = food.y < head.y
    food_down = food.y > head.y

    # Combine the state information
    state = [
        # Danger
        danger_left, danger_right, danger_up, danger_down,
        # Food direction
        food_left, food_right, food_up, food_down,
        # Current direction
        dir_left, dir_right, dir_up, dir_down
    ]

    return np.array(state, dtype=int)


def adjust_reward(game, done):
    if done:
        if game.score > 0:
            return 10  # Positive reward for eating food
        else:
            return -10  # Large negative reward for hitting a wall or itself
    return -1  # Small negative reward for each move

# Game configuration
game_config = {
    "w": 640,
    "h": 480,
}

# State and action dimensions
state_dim = (game_config["w"] // 20) * (game_config["h"] // 20)  # Assuming a grid representation
action_dim = 3  # For SnakeGameAI: [Turn Left, Go Straight, Turn Right]

# Agent configuration
agent_config = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "memory_size": 10000,
    "batch_size": 64,
    "lr": 1e-4,
    "epsilon_decay": 1e-3,
    "target_update": 100,
    "seed": 42,
    "gamma": 0.99,
    "alpha": 0.6,
    "beta": 0.4,
    "v_min": -10,
    "v_max": 10,
    "atom_size": 51,
}
total_reward=0
# Initialize game and agent
game = SnakeGameAI(w=game_config["w"], h=game_config["h"])
agent = RainbowAgent(**agent_config)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    state = game.reset()
    if state is None:
        raise ValueError("Game reset didn't return an initial state.")
    state = get_state(game)
    done = False  # Initialize 'done' to False at the start of each episode

    total_reward = 0

    # Game loop
    while not done:
        state = get_state(game)
        #print(f'State shape (before unsqueeze): {state.shape}')  # This should show (12,) for a 12-feature state vector
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        #print(f'State tensor shape (before forward pass): {state_tensor.shape}') # Expecting (1, 12)
        #print("device: ", device)
        #state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = agent.choose_action(state_tensor)

        reward, done, score = game.play_step(action)
        next_state = get_state(game)
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()
        total_reward += reward

        if next_state is None:
            raise ValueError("Game didn't return a next state.")

        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        state = next_state
        total_reward += reward

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    print(f'Episode: {episode}, Total Reward: {total_reward}, Score: {game.score}')

    # Save the model periodically
    if episode % 100 == 0:
        agent.save_model(f'model_episode_{episode}.pth')  # Implement save_model method in RainbowAgent

pygame.quit()
