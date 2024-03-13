import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

# Define directions and Point for the snake's movement and positioning
Direction = Enum('Direction', 'UP DOWN LEFT RIGHT')
Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
BLACK = (0, 0, 0)

BLOCK_SIZE = 10  # Decrease block size
w = 1280  # Increase width
h = 960  # Increase height
SPEED = 100


class SnakeGameAI:

    def __init__(self, w=1280, h=960):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w // 4, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        initial_size = 4  # Example: Start with a snake that has 5 blocks
        for i in range(1, initial_size):
            self.snake.append(Point(self.head.x - (i * BLOCK_SIZE), self.head.y))

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.state = self._get_initial_state()
        return self.state  # Make sure this returns the initial state, not None

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = pygame.font.SysFont('arial', 25).render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        """
        Create a simplified state representation of the game for AI training.
        """
        # Example: Create a binary matrix for the game area
        state = np.zeros((self.h // BLOCK_SIZE, self.w // BLOCK_SIZE), dtype=int)
        # Mark the snake's body in the matrix
        for point in self.snake:
            state[point.y // BLOCK_SIZE][point.x // BLOCK_SIZE] = 1
        # Mark the food
        if self.food:
            state[self.food.y // BLOCK_SIZE][self.food.x // BLOCK_SIZE] = 2
        # Flatten the matrix or create other representations as needed
        return state.flatten()

    def _get_initial_state(self):
        """
        Generate an initial state representation for the AI.
        This method should return the state in the format expected by your AI model.
        """
        state = np.zeros((self.h // BLOCK_SIZE, self.w // BLOCK_SIZE), dtype=int)
        # Mark the snake's body in the state array
        for point in self.snake:
            state[point.y // BLOCK_SIZE, point.x // BLOCK_SIZE] = 1
        # Mark the food
        if self.food:
            state[self.food.y // BLOCK_SIZE, self.food.x // BLOCK_SIZE] = 2
        # You can add more features to the state as needed

        # Flatten or process the state as required by your model
        # Example: Flatten the state array for a simple feedforward neural network
        return state.flatten()
