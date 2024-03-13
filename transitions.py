from collections import namedtuple

# Define the Transition named tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
