# 8-Puzzle Solver README

## General Description

This 8-Puzzle Solver is a Python application designed to solve the 8-puzzle game using various search algorithms. The 8-puzzle consists of a 3x3 grid with tiles numbered 1-8 and one blank space. The objective is to rearrange the tiles to a specific goal state. The application includes several algorithms and utility functions to facilitate the solving process.

### Algorithms Implemented
- Breadth-First Search (BFS)
- Iterative Deepening Depth-First Search (IDDFS)
- Greedy Best-First Search (GBFS)
- A* Search (AST)

### Main Functions
- `bfs`: Implements the Breadth-First Search algorithm.
- `iddfs`: Implements the Iterative Deepening Depth-First Search algorithm.
- `gbfs`: Implements the Greedy Best-First Search algorithm.
- `ast`: Implements the A* Search algorithm.
- `subNodes`: Generates possible moves from a given puzzle state.
- `move`: Executes a move in a given direction in the puzzle.
- `heuristic`: Calculates the heuristic value for GBFS and AST.

## State Space Representation

### States
Each state is represented as a list of integers, where each integer corresponds to a tile in the 8-puzzle grid. `0` represents the blank tile.

### Transitions
Transitions between states occur through tile movements - up, down, left, or right - contingent on the position of the blank tile.

## Heuristics for GBFS and A*

Heuristic functions are critical in informing search algorithms like GBFS (Greedy Best-First Search) and A* about the potential cost from a current state to the goal state. We have two custom heuristics in consideration:

### Tile Reversal Heuristic
The Tile Reversal Heuristic counts the number of direct adjacent tile reversals in the current state, with a reversal being defined as two tiles that are in the reverse order compared to their positions in the goal state. The heuristic is admissible because each reversal represents at least one move that must be made to correct the order, ensuring the heuristic never overestimates the true cost to reach the goal. It's also consistent (or monotonic) because the heuristic value decreases or remains unchanged with each move that corrects a reversal, satisfying the condition `h(N) <= cost(N, P) + h(P)` for each node N and successor P.

### Pattern Database Heuristic
The Pattern Database Heuristic is derived from a precomputed lookup table that maps specific tile configurations to their minimum move counts to reach the goal state. It is admissible since it provides the exact minimum cost to solve a subset of the puzzle, thus never overestimating. The heuristic is consistent as it inherently satisfies the monotonicity condition due to the nature of the database providing the exact minimum number of moves for its specific configuration.

## Admissibility and Consistency Proofs

### Admissibility Proof:
1. **Tile Reversal**: If a tile is part of a reversal, it must move at least once to reach its correct position, making the count of reversals an underestimate or an exact match of the true cost.
2. **Pattern Database**: Since the database contains actual minimum move counts for its configurations, it cannot overestimate the moves needed.

### Consistency Proof:
1. **Tile Reversal**: When a move is made, the heuristic value can only stay the same (if unrelated to reversals) or decrease (if it corrects a reversal). It never increases, thus fulfilling the consistency condition.
2. **Pattern Database**: By definition, the lookup table provides an exact count of moves for subsets of the puzzle, ensuring that the cost from the current node to a successor plus the cost from the successor to the goal is always equal to or more than the cost from the current node to the goal.

## Optimality of Algorithms

- **BFS and IDDFS**: These algorithms will find the optimal solution since they explore all possible paths without heuristic guidance.
- **GBFS**: The optimality of GBFS is not guaranteed, even with an admissible heuristic. Since GBFS does not consider the cost already incurred to reach a current state, it may choose a path that looks promising but leads to a longer solution. However, when using a pattern database as the heuristic, GBFS can become optimal. The pattern database provides an accurate and consistent estimation of the cost to reach the goal state, guiding GBFS towards the optimal path.
- **A***: A* is guaranteed to find an optimal solution when using admissible and consistent heuristics, as it considers both the cost so far and the estimated cost to the goal.

## Limitations of the Tile Reversal Heuristic in GBFS

The Tile Reversal Heuristic can sometimes mislead GBFS because it only considers the immediate number of reversals without accounting for the cumulative past cost. GBFS might opt for a state with fewer reversals but which is, in reality, further from the goal when considering the total path cost.

### Example:
Consider the following states of an 8-puzzle:

- Current State: 1 3 6 | 4 2 5 | 7 8 0 (Blank)
- Goal State: 0 1 2 | 3 4 5 | 6 7 8 (Blank)

Using the Tile Reversal Heuristic, the current state has two reversals (tiles 3 and 2, tiles 6 and 5). If GBFS encounters a state with one reversal but at a greater total path cost, it may still choose that state over others that would lead to an optimal path. This is because GBFS does not consider the total path cost, only the heuristic value, leading to suboptimal decisions.

## Running the Software

To run the software, follow these steps:
1. Ensure Python 3.x is installed on your system.
2. Run the command: `python algorithms.py "initial_board_state"`, where `initial_board_state` is a comma-separated string representing the initial puzzle configuration.

Example:
```bash
python algorithms.py "0,1,2,5,3,4,0,6,7,8"
```

## Output

The application outputs the following for each algorithm:
- The number of nodes expanded.
- The solution path as a sequence of moves.
- Additional statistics like cost, search depth, maximum search depth, and running time are also available in the code but commented out.

**Note**: A screenshot of the output is not included in this README but can be generated by running the application.

## Contributing

Contributions to the project are welcome. Please feel free to fork the repository and submit pull requests with any enhancements.
