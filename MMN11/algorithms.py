import argparse
from collections import deque
import math
import timeit

class PuzzleState:
    """
    This class represents a state of the puzzle game.
    """
    def __init__(self, state, parent, move, depth, cost, key):
        """
        Initialize a new state.
        """
        self.state = state  # The current state of the puzzle
        self.parent = parent  # The parent state that led to this state
        self.move = move  # The move that was made to get to this state
        self.depth = depth  # The depth of this state (number of moves made)
        self.cost = cost  # The cost of getting to this state
        self.key = key  # The key of this state
        if self.state:
            self.map = ''.join(str(e) for e in self.state)  # The map of this state

    def __eq__(self, other):
        """
        Check if this state is equal to another state.
        """
        return self.map == other.map

    def __lt__(self, other):
        """
        Check if this state is less than another state.
        """
        return self.map < other.map

    def __str__(self):
        """
        Return a string representation of this state.
        """
        return str(self.map)    

# Global variables
GoalState = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # The goal state of the puzzle
GoalNode = None  # The node that reaches the goal state
NodesExpanded = 0  # The total number of nodes that have been expanded
MaxSearchDeep = 0  # The maximum depth reached during the search
MaxFrontier = 0  # The maximum size of the frontier



def bfs(startState):
    """
    Perform a breadth-first search from the start state.
    """
    global MaxFrontier, GoalNode, MaxSearchDeep

    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    boardVisited = set()  # The set of states that have been visited
    Queue = deque([PuzzleState(startState, None, None, 0, 0, 0)])  # The queue of states to visit

    while Queue:
        node = Queue.popleft()  # Get the next state to visit
        nodeMapStr = ''.join(str(num) for num in node.state)
        boardVisited.add(nodeMapStr)  # Mark this state as visited

        if node.state == GoalState:  # If this state is the goal state
            GoalNode = node  # Save this state
            return Queue  # Return the queue of states to visit

        possiblePaths = subNodes(node, n)  # Get the possible next states
        for path in possiblePaths:
            pathMapStr = ''.join(str(num) for num in path.state)
            if pathMapStr not in boardVisited:  # If this state has not been visited
                Queue.append(path)  # Add it to the queue of states to visit
                boardVisited.add(pathMapStr)  # Mark this state as visited
                MaxSearchDeep = max(MaxSearchDeep, path.depth)  # Update the maximum depth

        MaxFrontier = max(MaxFrontier, len(Queue))  # Update the maximum size of the frontier

    return None  # If no solution is found, return None


def subNodes(node, n):
    """
    Generate the sub-nodes (children) of a given node.
    """
    global NodesExpanded
    NodesExpanded += 1  # Increment the count of nodes expanded

    nextPaths = []  # List to hold the generated sub-nodes
    index = node.state.index(0)  # Find the position of the empty space (0)

    # Up
    if index >= n:  # Ensure it's not in the top row
        newState = move(node.state, index, n, 1)  # direction 1 for up
        if newState is not None:
            # Create a new state and add it to the list of next paths
            nextPaths.append(PuzzleState(newState, node, 1, node.depth + 1, node.cost + 1, 0))

    # Down
    if index < n * (n - 1):  # Ensure it's not in the bottom row
        newState = move(node.state, index, n, 2)  # direction 2 for down
        if newState is not None:
            # Create a new state and add it to the list of next paths
            nextPaths.append(PuzzleState(newState, node, 2, node.depth + 1, node.cost + 1, 0))

    # Left
    if index % n != 0:  # Ensure it's not in the leftmost column
        newState = move(node.state, index, n, 3)  # direction 3 for left
        if newState is not None:
            # Create a new state and add it to the list of next paths
            nextPaths.append(PuzzleState(newState, node, 3, node.depth + 1, node.cost + 1, 0))

    # Right
    if index % n != n - 1:  # Ensure it's not in the rightmost column
        newState = move(node.state, index, n, 4)  # direction 4 for right
        if newState is not None:
            # Create a new state and add it to the list of next paths
            nextPaths.append(PuzzleState(newState, node, 4, node.depth + 1, node.cost + 1, 0))

    return nextPaths  # Return the list of sub-nodes



def move(state, index, n, direction):
    """
    Move the empty tile in the given direction and return the new state.
    """
    # Make a copy of the current state
    newState = state[:]

    # Calculate the row and column of the empty tile
    row, col = divmod(index, n)

    # Move up
    if direction == 1 and row > 0:
        # Swap the empty tile with the tile above it
        newState[index], newState[index - n] = newState[index - n], newState[index]
    
    # Move down
    elif direction == 2 and row < n - 1:
        # Swap the empty tile with the tile below it
        newState[index], newState[index + n] = newState[index + n], newState[index]

    # Move left
    elif direction == 3 and col > 0:
        # Swap the empty tile with the tile to the left of it
        newState[index], newState[index - 1] = newState[index - 1], newState[index]
    
    # Move right
    elif direction == 4 and col < n - 1:
        # Swap the empty tile with the tile to the right of it
        newState[index], newState[index + 1] = newState[index + 1], newState[index]

    # If the move is invalid, return None
    else:
        return None

    return newState  # Return the new state


def iddfs(startState):
    """
    Perform an iterative deepening depth-first search from the start state.
    """
    global MaxFrontier, GoalNode, MaxSearchDeep

    # Initialize the maximum depth, and other global variables
    MaxFrontier = MaxSearchDeep = 0
    GoalNode = None

    depth = 0
    while True:
        boardVisited = {}
        result = depth_limited_dfs(startState, depth, boardVisited)

        if result is not None:
            GoalNode = result
            return result  # Return the solution

        depth += 1
        MaxSearchDeep = max(MaxSearchDeep, depth)  # Update the maximum depth reached

def depth_limited_dfs(startState, depth_limit, visited):
    """
    Perform a depth-limited depth-first search from the start state.
    """
    stack = [PuzzleState(startState, None, None, 0, 0, 0)]
    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    while stack:
        node = stack.pop()  # Get the next node to visit

        if node.state == GoalState:
            return node  # Goal found

        if node.depth < depth_limit:  # If the depth limit has not been reached
            posiblePaths = reversed(subNodes(node,n))  # Get the possible next nodes
            for path in posiblePaths:
                if path.map not in visited or node.depth < visited[path.map]:
                    stack.append(path)  # Add the node to the stack
                    visited[path.map] = node.depth  # Mark the node as visited

    return None  # Goal not found within depth limit


def reverse_heuristic_gbfs(startState):
    """
    Perform a greedy best-first search from the start state.
    """
    global MaxFrontier, MaxSearchDeep, GoalNode

    # Determine the grid size (n x n) from the length of the start state
    n = int(len(startState) ** 0.5)

    # Generate the goal state for an n x n grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    # Initialize variables
    boardVisited = set()
    Queue = []
    
    # Convert the start state to a string format for heuristic calculation and storage
    startStateStr = ''.join(str(num) for num in startState)
    key = reverse_heuristic(startStateStr)  # Heuristic function now also takes the size of the grid

    # Add the initial state to the queue
    Queue.append(PuzzleState(startState, None, None, 0, 0, key))
    boardVisited.add(startStateStr)

    while Queue:
        # Sort the queue based on the heuristic value
        Queue.sort(key=lambda o: o.key)
        node = Queue.pop(0)  # Get the node with the lowest heuristic value

        # Check for goal state
        if node.state == GoalState:
            GoalNode = node
            return GoalNode  # Goal found

        # Expand the node and explore its children
        possiblePaths = subNodes(node, n)  # Ensure subNodes handles different grid sizes
        for path in possiblePaths:
            pathStr = ''.join(str(num) for num in path.state)
            if pathStr not in boardVisited:
                key = reverse_heuristic(pathStr)  # Update heuristic value for the child node
                path.key = key
                Queue.append(path)  # Add the node to the queue
                boardVisited.add(pathStr)  # Mark the node as visited
                MaxSearchDeep = max(MaxSearchDeep, path.depth)  # Update the maximum depth reached

    return None  # Goal not found


def pattern_database_gbfs(startState):
    """
    Perform a greedy best-first search from the start state.
    """
    global MaxFrontier, MaxSearchDeep, GoalNode

    # Determine the grid size (n x n) from the length of the start state
    n = int(len(startState) ** 0.5)

    # Generate the goal state for an n x n grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    # Initialize variables
    boardVisited = set()
    Queue = []
    
    # Convert the start state to a string format for heuristic calculation and storage
    startStateStr = ''.join(str(num) for num in startState)
    key = pattern_database_heuristic(startStateStr, pattern_database)  # Heuristic function now also takes the size of the grid

    # Add the initial state to the queue
    Queue.append(PuzzleState(startState, None, None, 0, 0, key))
    boardVisited.add(startStateStr)

    while Queue:
        # Sort the queue based on the heuristic value
        Queue.sort(key=lambda o: o.key)
        node = Queue.pop(0)  # Get the node with the lowest heuristic value

        # Check for goal state
        if node.state == GoalState:
            GoalNode = node
            return GoalNode  # Goal found

        # Expand the node and explore its children
        possiblePaths = subNodes(node, n)  # Ensure subNodes handles different grid sizes
        for path in possiblePaths:
            pathStr = ''.join(str(num) for num in path.state)
            if pathStr not in boardVisited:
                key = pattern_database_heuristic(pathStr, pattern_database)  # Update heuristic value for the child node
                path.key = key
                Queue.append(path)  # Add the node to the queue
                boardVisited.add(pathStr)  # Mark the node as visited
                MaxSearchDeep = max(MaxSearchDeep, path.depth)  # Update the maximum depth reached

    return None  # Goal not found



def ast(startState):
    """
    Perform an A* search from the start state.
    """
    global MaxFrontier, MaxSearchDeep, GoalNode

    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    # Initialize variables
    boardVisited = set()
    Queue = []

    # Convert the start state to a string format for heuristic calculation and storage
    startStateStr = ''.join(str(num) for num in startState)
    key = reverse_heuristic(startStateStr)  # Heuristic function now also takes the size of the grid

    # Add the initial state to the queue
    Queue.append(PuzzleState(startState, None, None, 0, 0, key))
    boardVisited.add(startStateStr)

    while Queue:
        # Sort the queue based on the key (heuristic + depth)
        Queue.sort(key=lambda o: o.key)
        node = Queue.pop(0)  # Get the node with the lowest key value

        # Check for goal state
        if node.state == GoalState:
            GoalNode = node
            return Queue  # Goal found

        # Expand the node and explore its children
        possiblePaths = subNodes(node, n)  # Ensure subNodes handles different grid sizes
        for path in possiblePaths:
            pathStr = ''.join(str(num) for num in path.state)
            if pathStr not in boardVisited:
                key = reverse_heuristic(pathStr)  # Update heuristic value for the child node
                path.key = key + path.depth  # A* key is heuristic value plus depth
                Queue.append(path)  # Add the node to the queue
                boardVisited.add(pathStr)  # Mark the node as visited
                MaxSearchDeep = max(MaxSearchDeep, path.depth)  # Update the maximum depth reached

    return None  # Goal not found

        
        

def reverse_heuristic(state):
    """
    Calculate the heuristic value for a given state based on the number of direct adjacent tile reversals.
    
    A reversal is defined as two tiles that are in the reverse order to where they should be in the goal state.
    
    :param state: The state to calculate the heuristic for.
    :return: The heuristic value.
    """
    # Initialize the heuristic value to 0
    heuristic_value = 0

    # Determine the size of the grid by taking the square root of the length of the state
    n = int(len(state) ** 0.5)

    # Check for reversals in rows
    for row in range(n):
        for col in range(n - 1):  # We subtract 1 because we're checking the current tile and the next one
            index = row * n + col  # Calculate the index of the current tile in the state list
            # If the current tile and the next one are a reversal and neither of them is the blank tile
            if state[index] > state[index + 1] and state[index] != 0 and state[index + 1] != 0:
                heuristic_value += 1  # Increment the heuristic value

    # Check for reversals in columns
    for col in range(n):
        for row in range(n - 1):  # We subtract 1 because we're checking the current tile and the next one
            index = row * n + col  # Calculate the index of the current tile in the state list
            # If the current tile and the one below it are a reversal and neither of them is the blank tile
            if state[index] > state[index + n] and state[index] != 0 and state[index + n] != 0:
                heuristic_value += 1  # Increment the heuristic value

    # Return the total number of reversals as the heuristic value
    return heuristic_value


def generate_pattern_database(goal):
    """
    Generate a pattern database for a given goal state.
    
    The pattern database is a dictionary where the keys are states and the values are the minimum number of moves
    required to reach the goal state from the key state. The database is generated using a breadth-first search.
    
    :param goal: The goal state.
    :return: The pattern database.
    """
    pattern_database = {tuple(goal): 0}
    queue = deque([goal])

    while queue:
        state = queue.popleft()
        cost = pattern_database[tuple(state)]

        for direction in range(1, 5):
            index = state.index(0)
            new_state = move(state, index, 3, direction)

            if new_state is not None and tuple(new_state) not in pattern_database:
                pattern_database[tuple(new_state)] = cost + 1
                queue.append(new_state)

    return pattern_database

# Generate the pattern database for the first 6 tiles
goal = [1, 2, 3, 4, 5, 6, 0, 0, 0]
pattern_database = generate_pattern_database(goal)

def pattern_database_heuristic(node, pattern_database):
    """
    Calculate the heuristic value for a given node.
    
    The heuristic value is the minimum number of moves required to reach the goal state from the node state,
    according to the pattern database. If the node state is not in the pattern database, a large number is returned.
    
    :param node: The node state.
    :param pattern_database: The pattern database.
    :return: The heuristic value.
    """
    node = tuple(node)

    if node in pattern_database:
        return pattern_database[node]

    return float('inf')



def main():
    """
    Main function to solve the puzzle.

    This function takes the initial board state as a command-line argument, checks if the number of elements is a perfect square,
    and then runs a list of algorithms (bfs, iddfs, gbfs, ast) on the initial state. It measures the time taken by each algorithm,
    and saves the total path result. The path is a list of moves that were made to reach the goal state from the initial state.
    """
    global GoalNode

    # Obtain information from calling parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('initialBoard')
    args = parser.parse_args()
    data = args.initialBoard.split(",")

    # Initialize InitialState
    InitialState = []

    # Check if the number of elements is a perfect square
    if math.sqrt(len(data)).is_integer():
        # Build initial board state
        InitialState = [int(num) for num in data]
    else:
        print("Invalid input. The number of elements should be a perfect square.")
        return

    # Define a list of algorithms
    algorithms = [bfs, iddfs, reverse_heuristic_gbfs, pattern_database_gbfs, ast]

    # For each algorithm
    for algorithm in algorithms:
        # Start operation
        start = timeit.default_timer()

        # Run the algorithm
        algorithm(InitialState)

        # Stop the timer
        stop = timeit.default_timer()
        time = stop - start

        # Save total path result
        deep = GoalNode.depth
        moves = []
        while InitialState != GoalNode.state:
            if GoalNode.move == 1:
                path = 'Up'
            if GoalNode.move == 2:
                path = 'Down'
            if GoalNode.move == 3:
                path = 'Left'
            if GoalNode.move == 4:
                path = 'Right'
            moves.insert(0, path)
            GoalNode = GoalNode.parent

        # Print results
        print(f"Algorithm: {algorithm.__name__}")
        print("nodes expanded: ", str(NodesExpanded))
        print("path: ", moves)
        # print("cost: ", len(moves))
        # print("search_depth: ", str(deep))
        # print("MaxSearchDeep: ", str(MaxSearchDeep))
        # print("running_time: ", format(time, '.8f'))

        # Generate output document for grade system
        # with open(f'{algorithm.__name__}_output.txt', 'w') as file:
        #     file.write("path_to_goal: " + str(moves) + "\n")
        #     file.write("cost_of_path: " + str(len(moves)) + "\n")
        #     file.write("nodes_expanded: " + str(NodesExpanded) + "\n")
        #     file.write("search_depth: " + str(deep) + "\n")
        #     file.write("max_search_depth: " + str(MaxSearchDeep) + "\n")
        #     file.write("running_time: " + format(time, '.8f') + "\n")

if __name__ == '__main__':
    main()