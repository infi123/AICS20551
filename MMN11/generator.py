import random
from algorithms import *

def random_state(steps):
    # Define the goal state and the possible moves
    state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    moves = [1, 2, 3, 4]

    # Make a number of random moves
    for _ in range(steps):
        # Choose a random move and try to apply it
        while True:
            chosen_move = random.choice(moves)
            index = state.index(0)
            new_state = move(state, index, 3, chosen_move)
            if new_state is not None:
                state = new_state
                break

    return state

# Generate a random state that is 6 to 10 steps away from the goal state
steps = random.randint(6, 10)
state = random_state(50)
print(state)