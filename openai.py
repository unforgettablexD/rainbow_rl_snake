import gym_snake_game
import gym

# both work
env = gym.make('Snake-v0', render_mode='ai')
env = gym_snake_game.make('Snake-v0', render_mode='human')
env.reset()

# for human playing
env.play()

# for ai playing
while True:
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    if done:
        break
env.close()