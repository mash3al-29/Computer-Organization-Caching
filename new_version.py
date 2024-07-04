import time
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd


# Comments on this project: Performance improves greatly as block size increases. However, increasing block size has
# diminishing returns on execution time decrease as the iteration overhead stops being the limiting factor of
# performance time Function to generate random matrices.This improvement is likely because larger blocks reduce the
# overhead associated with iterating over individual elements of the matrices. However, This diminishing returns
# occurs because once the block size reaches a certain threshold, the primary limiting factor in performance shifts
# from the iteration overhead to other factors, such as memory access patterns or cache behavior. Consequently,
# while increasing the block size initially leads to substantial performance gains, beyond a certain point,
# the reduction in execution time becomes less significant relative to the increase in block size. Therefore,
# finding the optimal block size involves balancing the reduction in iteration overhead with the diminishing returns
# on execution time decrease.

def generateMatrix(matSize, mat1, mat2):
    """
    Generate random matrices of size matSize.
    Args:
        matSize (int): Size of the matrices.
        mat1 (list): First matrix to be generated.
        mat2 (list): Second matrix to be generated.
    """
    for i in range(0, matSize):
        for j in range(0, matSize):
            mat1[i][j] = int(random.random() * 10)
            mat2[i][j] = int(random.random() * 10)


# Function to perform block matrix multiplication
def BlockMatTest(matSize, blockSize, A, B, C):
    """
    Perform block matrix multiplication.
    Args:
        matSize (int): Size of the matrices.
        blockSize (int): Size of the blocks.
        A (list): First matrix.
        B (list): Second matrix.
        C (list): Resultant matrix.
    Returns:
        list: Resultant matrix after multiplication.
    """
    if blockSize == 0:  # If block size is zero, perform normal matrix multiplication
        for i in range(0, matSize):
            for j in range(0, matSize):
                sum2 = 0
                for k in range(0, matSize):
                    sum2 = sum2 + A[i][k] * B[k][j]
                C[i][j] = sum2
        return C

    en = blockSize * (len(A) // blockSize)
    for i in range(0, en, blockSize):
        for j in range(0, en, blockSize):
            for ii in range(i, min(i + blockSize, matSize)):
                for jj in range(j, min(j + blockSize, matSize)):
                    sum2 = 0
                    for k in range(0, matSize):
                        sum2 = sum2 + A[ii][k] * B[k][jj]
                    C[ii][jj] = sum2
    return C


MatSizes = [100, 150, 200, 250, 300, 350, 400]
blockSizes = [0, 4, 8, 25, 32, 64, 85, 90]
data = []
# Loop through different matrix sizes
for matSize in MatSizes:
    mat1 = [[0 for x1 in range(matSize)] for y1 in range(matSize)]
    mat2 = [[0 for x2 in range(matSize)] for y2 in range(matSize)]
    mat3 = [[0 for x3 in range(matSize)] for y3 in range(matSize)]
    generateMatrix(matSize, mat1, mat2)
    # Measure time for different block sizes
    for blockSize in blockSizes:
        tic = time.perf_counter()
        BlockMatTest(matSize, blockSize, mat1, mat2, mat3)
        toc = time.perf_counter()
        data.append((matSize, blockSize, toc - tic))

# Create a DataFrame using the provided data with these column names
df = pd.DataFrame(data, columns=['matrix', 'block', 'time'])

# Create a new figure for plotting with a specified size
plt.figure(figsize=(10, 7))

# Iterate over each unique matrix size in the DataFrame
for matrix_size, row in df.groupby('matrix'):
    # Generate 400 evenly spaced numbers between the minimum and maximum block sizes in the current group
    x_new = np.linspace(row['block'].min(), row['block'].max(), 400)

    # Create a quadratic spline interpolation of the block size and time data points in the current group
    spLine = make_interp_spline(row['block'], row['time'], k=2)

    # Compute the interpolated time values for the new block size values
    newLine = spLine(x_new)

    # Plot the interpolated curve for the current matrix size
    plt.plot(x_new, newLine, label=f'Matrix Size {matrix_size}', linewidth=3, linestyle='-')

# Add labels and title to the plot and show it
plt.xlabel('Block')
plt.ylabel('Time')
plt.title('Time vs Block size')
plt.legend()
plt.show()
