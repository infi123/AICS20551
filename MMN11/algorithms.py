from collections import deque

def bfs(matrix):
    # Define the goal state
    goal = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

    # Define the start state
    start = matrix

    # Create a queue for BFS and enqueue the start state
    queue = deque([(start, [])])

    # Create a set to keep track of the explored nodes
    explored = set()

    while queue:
        # Dequeue a node from the queue
        state, path = queue.popleft()

        # Check if the current state is the goal state
        if state == goal:
            return {
                "name": "BFS",
                "nodes_opened": len(explored),
                "path": path,
            }

        # Mark the node as explored
        explored.add(str(state))

        # Get all possible next states
        next_states = get_next_states(state)

        for next_state, move in next_states:
            # If the node has not been explored yet, enqueue it
            if str(next_state) not in explored:
                queue.append((next_state, path + [move]))

    # If there's no solution, return None
    return None

def iddfs(matrix):
    # Implement Iterative Deepening DFS algorithm
    # Return the result as a dictionary
    return {
        "name": "IDDFS",
        "nodes_opened": 0,  # Replace with actual value
        "path": [],  # Replace with actual value
    }

def gbfs(matrix):
    # Implement Greedy Best First Search algorithm
    # Return the result as a dictionary
    return {
        "name": "GBFS",
        "nodes_opened": 0,  # Replace with actual value
        "path": [],  # Replace with actual value
    }

def a_star(matrix):
    # Implement A* algorithm
    # Return the result as a dictionary
    return {
        "name": "A*",
        "nodes_opened": 0,  # Replace with actual value
        "path": [],  # Replace with actual value
    }




def get_next_states(matrix):
    # Find the position of the 0 tile
    for i in range(4):
        for j in range(4):
            if matrix[i][j] == 0:
                x, y = i, j

    # Define the possible moves
    moves = [(x-1, y, 'U'), (x+1, y, 'D'), (x, y-1, 'L'), (x, y+1, 'R')]

    # Generate the next states
    next_states = []
    for x2, y2, move in moves:
        if 0 <= x2 < 4 and 0 <= y2 < 4:
            # Copy the current state to create a new state
            next_state = [row.copy() for row in matrix]
            # Swap the 0 tile with the adjacent tile
            next_state[x][y], next_state[x2][y2] = next_state[x2][y2], next_state[x][y]
            next_states.append((next_state, move))

    return next_states