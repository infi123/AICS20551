
# Introduction to Artificial Intelligence - 20551
## Maman 13 - README Overview

### Question 1 - Depth-First Search (DFS)
The Depth-First Search (DFS) algorithm has been implemented by utilizing a Stack, which inherently follows the Last-In-First-Out (LIFO) approach. This characteristic of the Stack ensures that the most recently added nodes are explored first, diving deep into the node paths before backtracking. The algorithm operates within a loop that continues to run as long as there are nodes to be explored in the Stack (i.e., the frontier is not empty).

In every iteration, the algorithm removes the latest node from the Stack to evaluate whether it meets the goal criteria. If the node corresponds to the goal state, the algorithm returns the path that led to this node. However, if the goal is not met, the algorithm proceeds to generate the node's successors. These successors are then added to the Stack, provided they have not been visited previously, ensuring that the search does not retrace its steps.

The discussion regarding the optimality of DFS is highlighted by using the mediumMaze example. DFS, by design, does not guarantee the most cost-effective path; it simply returns the first valid solution it encounters during its depth-centric search. Therefore, in scenarios where multiple solutions exist, the solution identified by DFS may not be the optimal one. This is particularly evident in the mediumMaze example where the cost of the solution found by DFS is significantly higher (130) than the solution found by BFS, which has a cost of 68. The BFS algorithm, which uses a Queue and follows a First-In-First-Out (FIFO) approach, is adept at finding the shortest path in terms of the number of steps taken, which is why the comparison is made to highlight the non-optimality of DFS.

In summary, while DFS is a powerful algorithm capable of traversing complex structures by exploring as far as possible along each branch before backtracking, its use-case is best suited for situations where finding any solution is the priority, rather than the optimal one.

### Question 2 - Breadth-First Search (BFS)
Breadth-First Search (BFS) is a strategy for searching in tree or graph data structures that expands all children of a node before moving to the next level. This approach is reflected in the implementation of BFS, which is similar to that of Depth-First Search (DFS) with a key difference: BFS uses a Queue for the frontier instead of a Stack. The Queue follows the First-In-First-Out (FIFO) principle, meaning nodes are explored in the order they were added.

An additional condition has been implemented before new successors are pushed onto the Queue. This condition checks if these successors have already been encountered and added to the frontier in previous iterations, thus preventing duplication and unnecessary expansions.

BFS is known for its capability to find the optimal solution. This characteristic is underpinned by its level-by-level examination of the search space. It methodically explores all possible paths within a given depth before proceeding to the next level. This exhaustive search ensures that once the goal is found, no other shorter path exists between the start node and the goal node, assuming that all paths have the same cost (i.e., the cost function is constant at 1). Hence, in environments where each move has a uniform cost, BFS is guaranteed to find the shortest path to the goal, which constitutes the optimal solution.

By scanning the entire breadth of the search space before going deeper, BFS can guarantee the shortest path, which is particularly useful in problems where all actions have the same cost.

### Question 3 - Uniform Cost Search (UCS)
Uniform Cost Search (UCS) is an algorithm used in searching that is also known as the cheapest first search. The implementation of UCS bears resemblance to the Breadth-First Search (BFS) algorithm in that it systematically explores paths from the starting node. However, unlike BFS which uses a simple Queue, UCS employs a PriorityQueue. This specialized queue orders items by cost, prioritizing the exploration of paths with the lowest cumulative cost first.

Each iteration within the UCS algorithm involves examining a node from the PriorityQueue. The algorithm ensures two key conditions are met before proceeding with a node's successor:

1. The successor has not been visited before.
2. The successor is not already present in the frontier, or if it is, the new path is more cost-effective than the previously recorded one.

If a successor node is already in the frontier with a higher path cost, UCS will update the frontier to reflect the better path. This step is critical as it reflects the algorithm's goal of not just reaching the target node but doing so with the least possible cost. Therefore, UCS is optimal for scenarios where path costs vary and finding the minimum-cost solution is required.

This approach allows UCS to be both complete and optimal, given that the cost of each step is positive. UCS will always find the least expensive path to the goal, making it well-suited for problems such as pathfinding on weighted graphs, where the edges have different costs associated with them.

### Question 4 - A* Search

The A* Search algorithm is implemented in a similar manner to the Depth-First Search (DFS) described in Question 1, with a significant difference in the frontier management. A* employs a `PriorityQueueWithFunction`, which prioritizes nodes based on a specific cost function. The cost function used in A* combines the actual cost from the start node to the current node (known as the `g` cost) with a heuristic estimate (`h`) of the cost to reach the goal. The heuristic employed here is the Manhattan distance, which calculates the cost based on the grid distance between the current node and the goal.

In the context of the `OpenMaze` problem, A* Search has been benchmarked against other search strategies, revealing its efficiency. Below are the performance statistics:

| Algorithm | Heuristic  | Cost | Expanded Nodes |
|-----------|------------|------|----------------|
| A*        | Manhattan  | 54   | 535            |
| UCS       | -          | 54   | 682            |
| BFS       | -          | 54   | 682            |
| DFS       | -          | 298  | 806            |

These results show that while A*, UCS, and BFS all find the optimal solution with the same cost, A* is the most efficient in terms of expanded nodes. This demonstrates the effectiveness of the A* algorithm, especially when the heuristic accurately reflects the remaining cost to the goal.

In summary, A* Search's use of a heuristic allows it to outperform other search methods by focusing the search in the most promising directions, thus reducing the number of expanded nodes and making it an efficient choice for the `OpenMaze` challenge.

### Question 5 - CornersProblem

In tackling the CornersProblem, the state is represented by a tuple that includes two elements:
1. `The position`: This represents the current position in the game environment.
2. `Visited corners`: A set that keeps track of all corners that have been visited.

The process for finding successors, implemented in the `getSuccessors` function, follows this logic for each possible action direction:
1. Confirm that moving in the current direction does not result in hitting a wall, thus ensuring valid movement.
2. If the new position reached is one of the unvisited corners, the set of visited corners is updated. This is done by creating a duplicate of the set for each direction and modifying it to include the newly visited corner.
3. Each valid successor is then added to the list of returned successors. A successor is defined by the new position, the action that led to it, and a uniform cost (in this case, a cost of 1).

This approach ensures that the agent explores the environment, aiming to visit all corners by updating its state to reflect the corners visited and avoiding invalid moves that would lead to collisions with walls.

### Question 6 - Corners Heuristic

For the CornersProblem, a specific heuristic called `cornersHeuristic` is employed to estimate the cost to reach the goal state from a given state. The heuristic strategy involves evaluating all possible permutations of paths to visit the remaining unvisited corners from the current position. The Manhattan distance is used to calculate the cost for each permutation, and the heuristic value is determined by the permutation with the lowest total distance.

For instance, if the top corners have not been visited, the heuristic will calculate the total distance for the following two permutations and select the minimum of these as the heuristic value:

1. The Manhattan distance from the current position to the top left corner, plus the Manhattan distance from the top left corner to the top right corner.
2. The Manhattan distance from the current position to the top right corner, plus the Manhattan distance from the top right corner to the top left corner.

This method ensures that the heuristic is both consistent and admissible, as it never overestimates the cost of reaching the goal. The use of the Manhattan distance as the heuristic measure is appropriate for grid-based problems where movements are restricted to horizontal and vertical paths.

### Question 7 - Food Heuristic

In addressing the FoodProblem, the heuristic chosen is a combination of Manhattan distances:

- The Manhattan distance to the closest piece of food.
- The Manhattan distance from the closest food to the farthest piece of food.

The heuristic function calculates these two distances and returns their sum. This approach ensures that the heuristic is consistent, as it always estimates the same cost when going from one state to another, and it is admissible because it never overestimates the actual cost of reaching the goal.

Performance metrics for this heuristic are noteworthy when applied to `trickySearch`, where it expanded 8178 nodes within 8 seconds. While this performance may seem slow, it is still an improvement over the Uniform Cost Search (UCS) in terms of both speed and the number of nodes expanded, as observed on my laptop during testing.

### Question 8 - Path to Closest Dot

The `findPathToClosestDot` function has been implemented using Breadth-First Search (BFS) on the agent's side. This strategy ensures that the algorithm will indeed find the nearest dot, considering that BFS examines all nodes at the present depth before moving on to nodes at the next level. This approach guarantees the closest dot is the first food dot encountered on the minimal path as BFS expands.

For the `AnyFoodSearchProblem`, there has been a specific modification to the `isGoalState` function. Instead of a predetermined goal state, the function now checks if there is any food present in the scanned node, allowing for dynamic goal state determination based on the presence of food.

However, it's important to note that the logic of finding the closest food does not always correlate with the shortest overall path to clear all points. For example, in a given scenario where there are multiple food dots, opting to go to the closest one first and then the next closest may result in a longer path overall compared to a path that may initially go to a farther dot but results in a shorter overall journey. This is illustrated in the provided example where the closest food logic (taking the path red -> blue -> green) takes 8 steps, while there exists a more efficient path (green -> red -> blue) that takes only 7 steps.

This nuance indicates that while BFS and the modified goal state check are efficient for finding the nearest dot, they may not always yield the most efficient path for collecting all food in the fewest steps possible.
