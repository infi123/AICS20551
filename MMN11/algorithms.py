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
    GoalState = list(range(1, n * n)) + [0]  # Generate the goal state dynamically

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

    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = list(range(1, n * n)) + [0]  # Generate the goal state dynamically

    depth = 0
    while True:
        visited = set()
        root = PuzzleState(startState, None, None, 0, 0, 0)
        result = dls(root, depth, visited, GoalState, n)
        if result is not None:
            GoalNode = result
            return visited
        depth += 1
        MaxSearchDeep = max(MaxSearchDeep, depth)
        MaxFrontier = max(MaxFrontier, len(visited))

def dls(node, depth, visited, GoalState, n):
    if node.map in visited or depth < 0:
        return None
    visited.add(node.map)
    if node.state == GoalState:
        return node
    elif depth == 0:
        return None
    else:
        for child in subNodes(node, n):  # Ensure subNodes handles different grid sizes
            result = dls(child, depth - 1, visited, GoalState, n)
            if result is not None:
                return result
    return None



def gbfs(startState):
    global MaxFrontier, MaxSearchDeep, GoalNode

    # Determine the grid size (n x n) from the length of the start state
    n = int(len(startState) ** 0.5)

    # Generate the goal state for an n x n grid
    GoalState = list(range(1, n * n)) + [0]

    # Initialize variables
    boardVisited = set()
    Queue = []
    
    # Convert the start state to a string format for heuristic calculation and storage
    startStateStr = ''.join(str(num) for num in startState)
    key = Heuristic(startStateStr, n)  # Heuristic function now also takes the size of the grid

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
                key = Heuristic(pathStr, n)  # Update heuristic value for the child node
                path.key = key
                Queue.append(path)
                boardVisited.add(pathStr)
                MaxSearchDeep = max(MaxSearchDeep, path.depth)

    return None


def ast(startState):
    global MaxFrontier, MaxSearchDeep, GoalNode

    n = int(len(startState) ** 0.5)  # Determine the size of the grid
    GoalState = list(range(1, n * n)) + [0]  # Generate the goal state dynamically

    # Initialize variables
    boardVisited = set()
    Queue = []

    # Convert the start state to a string format for heuristic calculation and storage
    startStateStr = ''.join(str(num) for num in startState)
    key = Heuristic(startStateStr, n)  # Heuristic function now also takes the size of the grid

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
                key = Heuristic(pathStr, n)  # Update heuristic value for the child node
                path.key = key + path.depth  # A* key is heuristic value plus depth
                Queue.append(path)
                boardVisited.add(pathStr)
                MaxSearchDeep = max(MaxSearchDeep, path.depth)

    return None

        
        


def Heuristic(node, n):
    goal = list(range(1, n * n)) + [0]  # Dynamically generate the goal state

    count = 0
    for i in range(len(node)):
        if int(node[i]) != goal[i]:
            # Calculate the correct row and column for the current tile
            correct_row = goal.index(int(node[i])) // n
            correct_col = goal.index(int(node[i])) % n

            # Calculate the current row and column for the current tile
            current_row = i // n
            current_col = i % n

            # Increment the count if the tile is not in the correct position
            if correct_row != current_row or correct_col != current_col:
                count += 1
    return count



def main():

    global GoalNode

    #a = [1,8,2,3,4,5,6,7,0]
    #point=Heuristic(a)
    #print(point)
    #return
    
    #info = "6,1,8,4,0,2,7,3,5" #20
    #info = "8,6,4,2,1,3,5,7,0" #26
    
    #Obtain information from calling parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('initialBoard')
    args = parser.parse_args()
    data = args.initialBoard.split(",")

    # Obtain information from calling parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
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


    #Start operation
    start = timeit.default_timer()

    function = args.method
    if(function=="bfs"):
        bfs(InitialState)
    if(function=="iddfs"):
        iddfs(InitialState) 
    if(function=="gbfs"):
        gbfs(InitialState) 
    if(function=="ast"):
        ast(InitialState) 

    stop = timeit.default_timer()
    time = stop-start

    #Save total path result
    deep=GoalNode.depth
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

    #'''
    #Print results
    print("path: ",moves)
    print("cost: ",len(moves))
    print("nodes expanded: ",str(NodesExpanded))
    print("search_depth: ",str(deep))
    print("MaxSearchDeep: ",str(MaxSearchDeep))
    print("running_time: ",format(time, '.8f'))
    #'''

    #Generate output document for grade system
    #'''
    file = open('output.txt', 'w')
    file.write("path_to_goal: " + str(moves) + "\n")
    file.write("cost_of_path: " + str(len(moves)) + "\n")
    file.write("nodes_expanded: " + str(NodesExpanded) + "\n")
    file.write("search_depth: " + str(deep) + "\n")
    file.write("max_search_depth: " + str(MaxSearchDeep) + "\n")
    file.write("running_time: " + format(time, '.8f') + "\n")
    file.close()
    #'''

if __name__ == '__main__':
    main()