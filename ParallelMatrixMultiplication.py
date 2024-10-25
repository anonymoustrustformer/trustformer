import numpy as np
from multiprocessing import Pool

def multiply_chunk(args):
    """Multiply a chunk of matrices A, B, and C."""
    chunk, A, B, C = args
    result_chunk = np.dot(np.dot(A[chunk[0]:chunk[1], :], B), C)
    return result_chunk

def parallel_matrix_multiplication(A, B, C, num_chunks):
    """Perform parallel matrix multiplication by dividing matrices into chunks."""
    result = np.zeros((A.shape[0], C.shape[1]))
    chunk_size = A.shape[0] // num_chunks
    
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    
    with Pool(num_chunks) as pool:
        chunk_results = pool.map(multiply_chunk, [(chunk, A, B, C) for chunk in chunks])
    
    for i, chunk in enumerate(chunks):
        result[chunk[0]:chunk[1], :] = chunk_results[i]
    
    return result

# Example usage:
if __name__ == "__main__":
    # Define matrices A, B, and C
    A = np.random.rand(6, 4)  # Matrix A of size (6, 4)
    B = np.random.rand(4, 6)  # Matrix B of size (4, 6)
    C = np.random.rand(6, 8)  # Matrix C of size (6, 8)

    # Number of chunks (i.e., processes)
    num_chunks = 3

    # Perform parallel matrix multiplication
    result = parallel_matrix_multiplication(A, B, C, num_chunks)

    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("Matrix C:")
    print(C)
    print("Result:")
    print(result)
