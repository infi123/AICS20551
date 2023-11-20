import sys

def get_matrix_from_args():
    # Check if exactly 16 command line arguments are provided
    if len(sys.argv) == 17:  # 16 arguments + 1 script name
        # Convert the command line arguments to integers
        numbers = [int(arg) for arg in sys.argv[1:]]
        # Check if numbers are in the range 0 to 15
        if all(0 <= num <= 15 for num in numbers):
            # Convert the list of numbers into a 4x4 matrix
            matrix = [numbers[i:i+4] for i in range(0, 16, 4)]
            return matrix
        else:
            print("All numbers should be in the range 0 to 15.")
            return None
    else:
        print("Exactly 16 numbers are required.")
        return None
    
