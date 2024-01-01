# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    from util import Stack

    frontier = Stack()  # For DFS we will use a Stack (LIFO)
    explored = []  # List of visited nodes
    node = problem.getStartState()
    path = []
    cost = 0

    # Checking if the start position is the goal
    if problem.isGoalState(node):
        return []

    # Pushing the start position to the frontier
    frontier.push((node, path, cost))

    while not frontier.isEmpty():
        node, path, cost = frontier.pop()
        # Add each visited node to the explored list
        explored.append(node)

        # Return if we got to goal
        if problem.isGoalState(node):
            return path

        # Per successor evaluate the full path to it and push to stack if not already explored (DFS)
        for child_node, child_direction, _ in problem.getSuccessors(node):
            child_path = path + [child_direction]
            child_cost = problem.getCostOfActions(child_path)

            if child_node not in explored:
                frontier.push((child_node, child_path, child_cost))

    # In case frontier was empty - the scan was completed, and we did not find a path - returning empty array
    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    from util import Queue

    frontier = Queue()  # For BFS we will use a Queue (FIFO)
    explored = [] # List of visited nodes
    node = problem.getStartState()
    path = []
    cost = 0

    # Checking if the start position is the goal
    if problem.isGoalState(node):
        return []

    # Pushing the start position to the frontier
    frontier.push((node, path, cost))

    while not frontier.isEmpty():
        node, path, cost = frontier.pop()

        # Add each visited node to the explored list
        explored.append(node)

        # Return if we got to goal
        if problem.isGoalState(node):
            return path

        # Per successor evaluate the full path to it and push to the queue
        for child_node, child_direction, _ in problem.getSuccessors(node):
            child_path = path + [child_direction]
            child_cost = problem.getCostOfActions(child_path)

            # Push only if not explored abd not already in queue
            if child_node not in explored and child_node not in (state[0] for state in frontier.list):
                frontier.push((child_node, child_path, child_cost))

    # In case frontier was empty - the scan was completed, and we did not find a path - returning empty array
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    from util import PriorityQueue

    # For UCS we will use a PriorityQueue (which returns the lower cost first by the definition in code)
    frontier = PriorityQueue()
    explored = []
    node = problem.getStartState()
    path = []
    cost = 0

    # Checking if the start position is the goal
    if problem.isGoalState(node):
        return []

    # Pushing the start position to the frontier
    frontier.push((node, path), cost)

    while not frontier.isEmpty():
        node, path = frontier.pop()

        # Add each visited node to the explored list
        explored.append(node)

        if problem.isGoalState(node):
            return path

        # Per successor evaluate the full path to it and push to the queue
        for child_node, child_direction, _ in problem.getSuccessors(node):
            child_path = path + [child_direction]
            child_cost = problem.getCostOfActions(child_path)

            # Push only if not explored and not already in queue
            if child_node not in explored and (child_node not in (state[2][0] for state in frontier.heap)):
                frontier.push((child_node, child_path), child_cost)

            # In case not explored but in queue - Update to the path with the lower cost
            elif child_node not in explored and (child_node in (state[2][0] for state in frontier.heap)):
                for state in frontier.heap:
                    if state[2][0] == child_node:
                        old_cost = problem.getCostOfActions(state[2][1])
                        if old_cost > child_cost:
                            frontier.update((child_node, child_path), child_cost)

    # In case frontier was empty - the scan was completed, and we did not find a path - returning empty array
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    from util import PriorityQueueWithFunction

    node_index = 0
    path_index = 1

    # For A* we will use a PriorityQueueWithFunction (which returns the lower "cost" that is provided by the function first - by the definition in code)
    frontier = PriorityQueueWithFunction(priorityFunction=lambda x: problem.getCostOfActions(x[path_index]) + heuristic(x[node_index], problem))
    explored = []
    node = problem.getStartState()
    path = []

    # Checking if the start position is the goal
    if problem.isGoalState(node):
        return []

    # Pushing the start position to the frontier
    frontier.push((node, path, heuristic))

    while not frontier.isEmpty():
        node, path, _ = frontier.pop()

        if node in explored:
            continue
        # Add new unvisited node to the explored list
        explored.append(node)

        if problem.isGoalState(node):
            return path

        # Per successor evaluate the full path to it and push to the queue
        for child_node, child_direction, _ in problem.getSuccessors(node):
            child_path = path + [child_direction]

            # Push only if not explored and not already in queue
            if child_node not in explored:
                frontier.push((child_node, child_path, heuristic))

    # In case frontier was empty - the scan was completed, and we did not find a path - returning empty array
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
