import argparse
from collections import deque
import math
import timeit


class PuzzleState:
    def __init__(self, state, parent, move, depth, cost, key):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost
        self.key = key
        if self.state:
            self.map = ''.join(str(e) for e in self.state)
    def __eq__(self, other):
        return self.map == other.map
    def __lt__(self, other):
        return self.map < other.map
    def __str__(self):
        return str(self.map)    

#Global variables***********************************************
GoalState = [0, 1, 2, 3, 4, 5, 6, 7, 8]
GoalNode = None # at finding solution
NodesExpanded = 0 #total nodes visited
MaxSearchDeep = 0 #max deep
MaxFrontier = 0 #max frontier



def bfs(startState):
    global MaxFrontier, GoalNode, MaxSearchDeep

    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    boardVisited = set()
    Queue = deque([PuzzleState(startState, None, None, 0, 0, 0)])

    while Queue:
        node = Queue.popleft()
        nodeMapStr = ''.join(str(num) for num in node.state)
        boardVisited.add(nodeMapStr)

        if node.state == GoalState:
            GoalNode = node
            return Queue

        possiblePaths = subNodes(node, n)  # Ensure subNodes handles different grid sizes
        for path in possiblePaths:
            pathMapStr = ''.join(str(num) for num in path.state)
            if pathMapStr not in boardVisited:
                Queue.append(path)
                boardVisited.add(pathMapStr)
                MaxSearchDeep = max(MaxSearchDeep, path.depth)

        MaxFrontier = max(MaxFrontier, len(Queue))

    return None


def subNodes(node, n):
    global NodesExpanded
    NodesExpanded += 1

    nextPaths = []
    index = node.state.index(0)  # Find the position of the empty space (0)

    # Up
    if index >= n:  # Ensure it's not in the top row
        newState = move(node.state, index, n, 1)  # direction 1 for up
        if newState is not None:
            nextPaths.append(PuzzleState(newState, node, 1, node.depth + 1, node.cost + 1, 0))

    # Down
    if index < n * (n - 1):  # Ensure it's not in the bottom row
        newState = move(node.state, index, n, 2)  # direction 2 for down
        if newState is not None:
            nextPaths.append(PuzzleState(newState, node, 2, node.depth + 1, node.cost + 1, 0))

    # Left
    if index % n != 0:  # Ensure it's not in the leftmost column
        newState = move(node.state, index, n, 3)  # direction 3 for left
        if newState is not None:
            nextPaths.append(PuzzleState(newState, node, 3, node.depth + 1, node.cost + 1, 0))

    # Right
    if index % n != n - 1:  # Ensure it's not in the rightmost column
        newState = move(node.state, index, n, 4)  # direction 4 for right
        if newState is not None:
            nextPaths.append(PuzzleState(newState, node, 4, node.depth + 1, node.cost + 1, 0))

    return nextPaths



def move(state, index, n, direction):
    # Make a copy of the current state
    newState = state[:]

    # Calculate the row and column of the empty tile
    row, col = divmod(index, n)

    # Move up
    if direction == 1 and row > 0:
        newState[index], newState[index - n] = newState[index - n], newState[index]
    
    # Move down
    elif direction == 2 and row < n - 1:
        newState[index], newState[index + n] = newState[index + n], newState[index]

    # Move left
    elif direction == 3 and col > 0:
        newState[index], newState[index - 1] = newState[index - 1], newState[index]
    
    # Move right
    elif direction == 4 and col < n - 1:
        newState[index], newState[index + 1] = newState[index + 1], newState[index]

    # If the move is invalid, return None
    else:
        return None

    return newState


def iddfs(startState):
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
        MaxSearchDeep = max(MaxSearchDeep, depth)

def depth_limited_dfs(startState, depth_limit, visited):
    stack = [PuzzleState(startState, None, None, 0, 0, 0)]
    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    while stack:
        node = stack.pop()

        if node.state == GoalState:
            return node  # Goal found

        if node.depth < depth_limit:
            posiblePaths = reversed(subNodes(node,n))
            for path in posiblePaths:
                if path.map not in visited or node.depth < visited[path.map]:
                    stack.append(path)
                    visited[path.map] = node.depth

    return None  # Goal not found within depth limit


def gbfs(startState):
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
    key = Heuristic(startStateStr, pattern_database)  # Heuristic function now also takes the size of the grid

    # Add the initial state to the queue
    Queue.append(PuzzleState(startState, None, None, 0, 0, key))
    boardVisited.add(startStateStr)

    while Queue:
        # Sort the queue based on the heuristic value
        Queue.sort(key=lambda o: o.key)
        node = Queue.pop(0)

        # Check for goal state
        if node.state == GoalState:
            GoalNode = node
            return GoalNode

        # Expand the node and explore its children
        possiblePaths = subNodes(node, n)  # Ensure subNodes handles different grid sizes
        for path in possiblePaths:
            pathStr = ''.join(str(num) for num in path.state)
            if pathStr not in boardVisited:
                key = Heuristic(pathStr, pattern_database)  # Update heuristic value for the child node
                path.key = key
                Queue.append(path)
                boardVisited.add(pathStr)
                MaxSearchDeep = max(MaxSearchDeep, path.depth)

    return None


def ast(startState):
    global MaxFrontier, MaxSearchDeep, GoalNode

    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = [0] + list(range(1, n * n))  # Generate the goal state dynamically

    # Initialize variables
    boardVisited = set()
    Queue = []

    # Convert the start state to a string format for heuristic calculation and storage
    startStateStr = ''.join(str(num) for num in startState)
    key = Heuristic(startStateStr, pattern_database)  # Heuristic function now also takes the size of the grid

    # Add the initial state to the queue
    Queue.append(PuzzleState(startState, None, None, 0, 0, key))
    boardVisited.add(startStateStr)

    while Queue:
        # Sort the queue based on the key (heuristic + depth)
        Queue.sort(key=lambda o: o.key)
        node = Queue.pop(0)

        # Check for goal state
        if node.state == GoalState:
            GoalNode = node
            return Queue

        # Expand the node and explore its children
        possiblePaths = subNodes(node, n)  # Ensure subNodes handles different grid sizes
        for path in possiblePaths:
            pathStr = ''.join(str(num) for num in path.state)
            if pathStr not in boardVisited:
                key = Heuristic(pathStr, pattern_database)  # Update heuristic value for the child node
                path.key = key + path.depth  # A* key is heuristic value plus depth
                Queue.append(path)
                boardVisited.add(pathStr)
                MaxSearchDeep = max(MaxSearchDeep, path.depth)

    return None

        
        

def generate_pattern_database(goal):
    # Initialize the pattern database and the queue
    pattern_database = {tuple(goal): 0}
    queue = deque([goal])

    while queue:
        state = queue.popleft()
        cost = pattern_database[tuple(state)]

        # Generate all possible next states
        for direction in range(1, 5):
            index = state.index(0)
            new_state = move(state, index, 3, direction)

            # If the new state is valid and not already in the database, add it
            if new_state is not None and tuple(new_state) not in pattern_database:
                pattern_database[tuple(new_state)] = cost + 1
                queue.append(new_state)

    return pattern_database

# Generate the pattern database for the first 6 tiles
goal = [1, 2, 3, 4, 5, 6, 0, 0, 0]
pattern_database = generate_pattern_database(goal)

def Heuristic(node, pattern_database):
    # Convert the node to a tuple to use it as a key in the pattern database
    node = tuple(node)

    # If the node is in the pattern database, return the stored value
    if node in pattern_database:
        return pattern_database[node]

    # If the node is not in the pattern database, return a large number
    # This should not happen if the pattern database is complete
    return float('inf')



def main():

    global GoalNode

    
    #Obtain information from calling parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('method')
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
    algorithms = [bfs, iddfs, gbfs, ast]

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
        print("path: ", moves)
        print("cost: ", len(moves))
        print("nodes expanded: ", str(NodesExpanded))
        print("search_depth: ", str(deep))
        print("MaxSearchDeep: ", str(MaxSearchDeep))
        print("running_time: ", format(time, '.8f'))

        # Generate output document for grade system
        with open(f'{algorithm.__name__}_output.txt', 'w') as file:
            file.write("path_to_goal: " + str(moves) + "\n")
            file.write("cost_of_path: " + str(len(moves)) + "\n")
            file.write("nodes_expanded: " + str(NodesExpanded) + "\n")
            file.write("search_depth: " + str(deep) + "\n")
            file.write("max_search_depth: " + str(MaxSearchDeep) + "\n")
            file.write("running_time: " + format(time, '.8f') + "\n")

if __name__ == '__main__':
    main()