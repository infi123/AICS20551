# Question 1 - Reflex Agent Enhancement

To enhance the reflex agent, a new evaluation function has been developed with the following assumptions:

- Remaining stationary is highly detrimental, as it has been observed to improve scores and prevent deadlock situations.
- Consuming food as part of the current action is considered beneficial.
- A large distance from the nearest food source is unfavorable, as it is desired to minimize this distance to progress towards victory.
- Proximity to an active ghost is considered extremely negative, indicating that avoidance is crucial.
- Being near a frightened ghost is seen as advantageous, as the goal is to consume the ghost.

Based on these principles, the evaluation function is constructed as follows:

1. If the action is to stop, the evaluation is set to negative infinity.
2. Presence of an active ghost within a Manhattan distance of less than 2 also results in a negative infinity evaluation.
3. In other cases:
   - The evaluation starts with the score of the current state.
   - Eating food in the current move adjusts the evaluation by ±100.
   - Remaining food is accounted for by subtracting the Manhattan distance to the nearest food from the evaluation, which encourages closing the distance to food sources.
   - Encountering a scared ghost changes the evaluation by ±100.

# Question 2 - MiniMax Algorithm

The MiniMax algorithm has been implemented using three primary methods:

1. **minimax(state, depth, agent)**
   - The central recursive method for implementing the minimax strategy.
   - If the state is terminal (i.e., a winning or losing state) or if `depth` equals 0 or there are no further legal actions, return the state's evaluation.
   - Otherwise, for the current state and depth:
     - If the agent is Pacman (`agent == 0`), invoke `maxValue`.
     - For ghost agents, invoke `minValue`.

2. **maxValue(state, depth, agent)**
   - **Overview:**
     - Initialize the maximum value to negative infinity.
     - If no legal actions exist, return the evaluation of the current state.
     - For each legal action and successor state:
       1. Call `minimax` for the successor state (which will call `minValue` subsequently).
       2. If the value returned is greater than the current maximum, update the max value and the corresponding action.
     - Return the optimal max value and action.

3. **minValue(state, depth, agent)**
   - **Overview:**
     - Initialize the minimum value to positive infinity.
     - If no legal actions exist, return the evaluation of the current state.
     - For each legal action, successor state, and ghost agent:
       1. Call `minimax` for the next agent.
          - If the next agent is a ghost, depth remains unchanged.
          - If it's the last ghost, call `minimax` with `depth - 1` and Pacman as the agent (`agent = 0`).
       2. If the returned value is lower than the current minimum, update the min value and the corresponding action.
     - Return the optimal min value and action.

The `getAction` method triggers the `minimax` function with the current state, the desired depth (limited by resources in some scenarios), and identifies Pacman as the agent. Finally, the calculated minimax values and suggested actions are returned.

# Question 3 - Alpha Beta Pruning

The MiniMax algorithm has been extended with alpha-beta pruning using three main functions, enhancing the approach described in Question 2:

1. **alphaBetaPruning(state, depth, agent, alpha, beta)**
   - This principal recursive function accesses the minimax algorithm with alpha-beta pruning enhancements. It operates similarly to the minimax function but includes additional alpha and beta parameters for pruning.

2. **maxValue(state, depth, agent, alpha, beta)**
   - Functions like the minimax `maxValue`, with these augmentations:
     - If `maxValue > beta`, the function prunes and returns the current `maxValue` and corresponding `maxAction`.
     - Update alpha to the maximum value if `alpha < maxValue` for each action.

3. **minValue(state, depth, agent, alpha, beta)**
   - Mirrors the minimax `minValue`, with these augmentations:
     - If `minValue < alpha`, the function prunes and returns the current `minValue` and corresponding `minAction`.
     - Update beta to the minimum value if `beta > minValue` for each action.

The `getAction` method initiates the `alphaBetaPruning` with the current state, desired depth, agent, and initial alpha and beta values, eventually yielding the optimized minimax values and recommended actions after pruning.

# Question 4 - Expectimax

The implementation of the Expectimax algorithm is carried out using three methods, analogous to the minimax algorithm detailed in question 2:

1. **expectiMax(state, depth, agent)**
   - This function serves as the main recursive mechanism of the Expectimax algorithm. It's identical to the primary recursive function of minimax, with the distinction that for ghost agents, the `expValue` function is invoked instead of `minValue`.

2. **maxValue(state, depth, agent)**
   - The code and logic are identical to the `maxValue` function of the minimax algorithm, without any modifications.

3. **expValue(state, depth, agent)**
   - This function is modified from the minimax `minValue` function. The key differences are:
     - Instead of computing a minimum value, the algorithm:
       1. Calculates the value for each possible action.
       2. Computes the average of all possible actions' values to determine the expected utility rather than the minimum of all possible options.

# Question 5 - Better Evaluation Function

Multiple iterations were undertaken to formulate an effective evaluation function that achieves a full score in the autograder.

The evaluation criteria are based on several key points:

1. The current game score.
2. Whether the state is a winning or losing condition.
3. The remaining number of food pellets, with fewer being better.
4. The distance to the nearest food pellet, with closer being better.
5. The proximity to active or scared ghosts, with the former being a negative factor and the latter a positive one.

Each element contributes to the evaluation function, assigning positive values to beneficial moves and negative values to detrimental ones.

The evaluation function was constructed as follows:

1. If the game state signals a victory or defeat, assign an extremely high or low value, respectively.
2. Otherwise, compute the score by considering:
   - A base evaluation: `5 * (current game score)` to incentivize score-boosting moves.
   - Adjusting the evaluation by `-10 * (remaining food pellets)` to promote food consumption and favor states with less remaining food.
   - Subtracting `10` if any food remains, as an incentive to consume the last food pellet and avoid situations where Pacman gets trapped.
   - Modifying the evaluation by `+10 / (sum of distances to the three closest foods)` and `+5 / (sum of distances to the five closest foods)` to encourage proximity to food.
   - Adjusting the evaluation by `-10 / (distance to the nearest active ghost)` to discourage close encounters with active ghosts.
   - Adding `+10 / (distance to the nearest scared ghost)` to encourage chasing scared ghosts.

The refined evaluation function prioritizes moves that lead to winning scenarios, optimal food consumption, and strategic ghost interaction.
