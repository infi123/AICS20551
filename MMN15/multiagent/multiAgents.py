# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** My code starts here ***"

        # Never stop
        if action == Directions.STOP:
            return float('-inf')

        # Start with game score as base and modify evaluation
        evaluation = successorGameState.getScore()

        # Food considerations in evaluation:
        oldFoodList = currentGameState.getFood().asList()
        newFoodList = newFood.asList()

        # If we ate food in the move, add 100 to the evaluation
        if len(oldFoodList) > len(newFoodList):
            evaluation += 100

        # Get the closest food distance (using Manhattan distance)
        if len(newFoodList) > 0:
            closestFood = min([manhattanDistance(newPos, food) for food in newFoodList])
        else:
            closestFood = 0

        # In case the closest food is too far, we don't want to go there so we give it a negative score in the evaluation
        evaluation -= closestFood

        # Get the closest ghost distance (using Manhattan distance)
        ghostList = successorGameState.getGhostPositions()
        closestGhost = min([manhattanDistance(newPos, ghost) for ghost in ghostList])

        # If the ghost is too close, return a negative evaluation
        if closestGhost < 2:
            return float('-inf')

        # If the ghost is scared, return a positive evaluation (we can eat it!)
        if newScaredTimes[0] > 0:
            evaluation += 100

        return evaluation


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** My code starts here ***"

        # Evaluate the correct action using the minimax algorithm
        action = self.minimax(state=gameState,
                              depth=self.depth,
                              agent=0)[1]

        return action

    def minimax(self, state, depth, agent):
        # If we reached the end of the game, or no legal actions, or max depth of minimax return the evaluation
        if state.isWin() or state.isLose() or depth == 0 or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        # If we are at pacman, we want to maximize
        if agent == 0:
            return self.maxValue(state, depth, agent)
        # If we are at a ghost, we want to minimize
        else:
            return self.minValue(state, depth, agent)

    def maxValue(self, state, depth, agent):

        # Get legal actions for pacman
        legalActions = state.getLegalActions(agent)

        # If there are no legal actions, return the evaluation
        if len(legalActions) == 0:
            return self.evaluationFunction(state), None

        # Initialize the maximum value to the lowest possible value
        maxVal = float('-inf')
        maxAction = None

        # For each legal action, get the value of the successor state
        for action in legalActions:
            successorState = state.generateSuccessor(agent, action)


            # Get the value of the successor state per ghost
            value = self.minimax(successorState, depth, agent + 1)[0]

            # If the value is higher than the current maximum, update the maximum value and action
            if value > maxVal:
                maxVal = value
                maxAction = action

        return maxVal, maxAction

    def minValue(self, state, depth, agent):

        # Get legal actions for ghost agent
        legalActions = state.getLegalActions(agent)

        # If there are no legal actions, return the evaluation
        if len(legalActions) == 0:
            return self.evaluationFunction(state), None

        # Initialize the minimum value to the highest possible value
        minVal = float('inf')
        minAction = None

        # For each legal action, get the value of the successor state
        for action in legalActions:
            successorState = state.generateSuccessor(agent, action)

            # Get ghosts number
            ghostNumber = state.getNumAgents() - 1

            # If we are at the last ghost, we want to go back to pacman
            if agent == ghostNumber:
                # Get the value of the successor state
                value = self.minimax(successorState, depth - 1, 0)[0]
            # If we are not at the last ghost, we want to go to the next ghost
            else:
                # Get the value of the successor state
                value = self.minimax(successorState, depth, agent + 1)[0]

            # If the value is less than the current minimum, update the minimum value and action
            if value < minVal:
                minVal = value
                minAction = action

        return minVal, minAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** My code starts here ***"

        # Evaluate the correct action using the minimax algorithm
        action = self.alphaBetaPruning(state=gameState,
                                       depth=self.depth,
                                       agent=0,
                                       alpha=float('-inf'),
                                       beta=float('inf'))[1]

        return action

    def alphaBetaPruning(self, state, depth, agent, alpha, beta):
        # If we reached the end of the game, or no legal actions, or max depth of minimax return the evaluation
        if state.isWin() or state.isLose() or depth == 0 or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        # If we are at pacman, we want to
        if agent == 0:
            return self.maxValue(state, depth, agent, alpha, beta)
        # If we are at a ghost, we want to minimize
        else:
            return self.minValue(state, depth, agent, alpha, beta)

    def maxValue(self, state, depth, agent, alpha, beta):

        # Get legal actions for pacman
        legalActions = state.getLegalActions(agent)

        # Initialize the maximum value to the lowest possible value
        maxVal = float('-inf')
        maxAction = None

        # For each legal action, get the value of the successor state
        for action in legalActions:
            successorState = state.generateSuccessor(agent, action)

            # Get the value of the successor state per ghost
            value = self.alphaBetaPruning(successorState, depth, agent + 1, alpha, beta)[0]

            # If the value is higher than the current maximum, update the maximum value and action
            if value > maxVal:
                maxVal = value
                maxAction = action

            # If the value is higher than beta, we can prune
            if maxVal > beta:
                return maxVal, maxAction

            # Update alpha
            alpha = max(alpha, maxVal)

        return maxVal, maxAction

    def minValue(self, state, depth, agent, alpha, beta):

        # Get legal actions for ghost agent
        legalActions = state.getLegalActions(agent)

        # Initialize the minimum value to the highest possible value
        minVal = float('inf')
        minAction = None

        # For each legal action, get the value of the successor state
        for action in legalActions:
            successorState = state.generateSuccessor(agent, action)

            # Get ghosts number
            ghostNumber = state.getNumAgents() - 1

            # If we are at the last ghost, we want to go back to pacman
            if agent == ghostNumber:
                # Get the value of the successor state
                value = self.alphaBetaPruning(successorState, depth - 1, 0, alpha, beta)[0]
            # If we are not at the last ghost, we want to go to the next ghost
            else:
                # Get the value of the successor state
                value = self.alphaBetaPruning(successorState, depth, agent + 1, alpha, beta)[0]

            # If the value is less than the current minimum, update the minimum value and action
            if value < minVal:
                minVal = value
                minAction = action

            # If the value is less than alpha, we can prune
            if minVal < alpha:
                return minVal, minAction

            # Update beta
            beta = min(beta, minVal)

        return minVal, minAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** My code starts here ***"

        # Evaluate the correct action using the minimax algorithm
        action = self.expectiMax(state=gameState,
                                 depth=self.depth,
                                 agent=0)[1]

        return action

    def expectiMax(self, state, depth, agent):

        # If we reached the end of the game, or no legal actions, or max depth of minimax return the evaluation
        if state.isWin() or state.isLose() or depth == 0 or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        # If we are at pacman, we want to
        if agent == 0:
            return self.maxValue(state, depth, agent)
        # If we are at a ghost, we want to return the expected value
        else:
            return self.expValue(state, depth, agent)

    def maxValue(self, state, depth, agent):

        # Get legal actions for pacman
        legalActions = state.getLegalActions(agent)

        # Initialize the maximum value to the lowest possible value
        maxVal = float('-inf')
        maxAction = None

        # For each legal action, get the value of the successor state
        for action in legalActions:
            successorState = state.generateSuccessor(agent, action)

            # Get the value of the successor state per ghost
            value = self.expectiMax(successorState, depth, agent + 1)[0]

            # If the value is higher than the current maximum, update the maximum value and action
            if value > maxVal:
                maxVal = value
                maxAction = action

        return maxVal, maxAction

    def expValue(self, state, depth, agent):

        # Get legal actions for ghost agent
        legalActions = state.getLegalActions(agent)
        probability = 1.0 / len(legalActions)

        # Initialize the minimum value to the highest possible value
        expVal = 0
        expAction = None

        # For each legal action, get the value of the successor state
        for action in legalActions:
            successorState = state.generateSuccessor(agent, action)

            # Get ghosts number
            ghostNumber = state.getNumAgents() - 1

            # If we are at the last ghost, we want to go back to pacman
            if agent == ghostNumber:
                # Get the value of the successor state
                value = probability * self.expectiMax(successorState, depth - 1, 0)[0]
            # If we are not at the last ghost, we want to go to the next ghost
            else:
                # Get the value of the successor state
                value = probability * self.expectiMax(successorState, depth, agent + 1)[0]

            # Update the current expectation value
            expVal += value

        return expVal, expAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    1. In case the game state is win or loose, return a corresponding very high / low value
    2. Otherwise, evaluate the score based on the following:
        2.1. Base evaluation = 5 * <the current game score>
        2.2. evaluation += -10 * <The number of food left to eat>
        2.3. evaluation += -10 if there is food to eat
        2.4. evaluation += +(10. / <Sum of distance to top 3 closest foods>) + (5. / <Sum of distance to top 5 closest foods>)
        2.5. evaluation += -10 * (1. / <Distance to closest active ghost>)
        2.6. evaluation += +10 * (1. / <Distance to closest scared ghost>)
    """
    "*** My code starts here ***"

    # High score for win state
    if currentGameState.isWin():
        return 100000.

    # Low score for win state
    if currentGameState.isLose():
        return -100000.


    foodList = currentGameState.getFood().asList()
    currentPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    # Create arrays for active and scared ghosts
    activeGhosts = []
    scaredGhosts = []

    # Find active and scared ghosts
    for ghost in ghostStates:
        if ghost.scaredTimer:  # Is scared ghost
            scaredGhosts.append(ghost.getPosition())
        else:
            activeGhosts.append(ghost.getPosition())

    # Start with multiplied game score as base and modify evaluation
    evaluation = 5 * currentGameState.getScore()


    # Update evaluation based on left food to eat, we want to punish for not eating food
    evaluation += - 10 * len(foodList)


    # Update evaluation based on food distances
    foodDistances = [manhattanDistance(currentPosition, food) for food in foodList]
    sortedFoodsDistances = sorted(foodDistances)

    closeFoodsSum = sum(sortedFoodsDistances[-5:])
    closestFoodsSum = sum(sortedFoodsDistances[-3:])

    # If there is food to each, decrease the evaluation
    if len(sortedFoodsDistances) > 0:
        evaluation += - 10.

    # We want to lower the evaluation for any close foods based on distance
    # Very close foods lower the score more than normal close foods
    evaluation += (5. / closeFoodsSum) + (10. / closestFoodsSum)


    # Update evaluation based on active ghost distances
    # It is very bad for pacman to have close active ghosts. We want him to avoid them.
    activeGhostDistances = [manhattanDistance(currentPosition, ghost) for ghost in activeGhosts]

    if len(activeGhostDistances) > 0:
        evaluation += - 10. / min(activeGhostDistances)

    # Update evaluation based on scared ghost distances
    # It is very good for pacman to have close scared ghosts. We want him to eat them.
    scaredGhostDistances = [manhattanDistance(currentPosition, ghost) for ghost in scaredGhosts]

    if len(scaredGhostDistances) > 0:
        evaluation += + 10. / min(scaredGhostDistances)

    return evaluation


# Abbreviation
better = betterEvaluationFunction
