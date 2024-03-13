import pygame
from snake_game_ai import SnakeGameAI  # Make sure this is correctly imported
from rainbow_agent import RainbowAgent  # Verify this matches your file and class name

# Initialize Pygame
pygame.init()

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
    done = False  # Initialize 'done' to False at the start of each episode

    total_reward = 0

    # Game loop
    while not done:
        action = agent.choose_action(state)
        # Make sure game.play_step() and game.get_state() correctly handle the state
        reward, done, _ = game.play_step(action)
        next_state = game.get_state()

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
