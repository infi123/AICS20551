import utils
import algorithms

def main():
    matrix = utils.get_matrix_from_args()
    if matrix is not None:
        print("Received matrix:")
        for row in matrix:
            print(row)
        
        # Run the algorithms
        for algorithm in [algorithms.bfs, algorithms.iddfs, algorithms.gbfs, algorithms.a_star]:
            result = algorithm(matrix)
            print(f"\nAlgorithm: {result['name']}")
            print(f"Nodes opened: {result['nodes_opened']}")
            print(f"Path: {result['path']}")

if __name__ == "__main__":
    main()