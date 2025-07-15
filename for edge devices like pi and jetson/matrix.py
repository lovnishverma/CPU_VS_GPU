import numpy as np
import time

def benchmark_matrix_multiplication(size=1000):
    print(f"Running matrix multiplication benchmark with size {size}x{size}...")
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    start_time = time.time()
    C = np.dot(A, B)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")
    return elapsed_time

if __name__ == "__main__":
    benchmark_matrix_multiplication(size=1000)
