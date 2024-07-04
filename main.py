import numpy as np
import time
import random
import matplotlib.pyplot as plt


def BlockMatTest(matSize, blockSize, A, B, C):
    if blockSize == 0:  # If block size is zero, perform normal matrix multiplication
        for i in range(matSize):
            for j in range(matSize):
                for k in range(matSize):
                    C[i, j] += A[i, k] * B[k, j]

        # print('0 block')
        # print('\n')
        # print(A)
        # print('\n')
        # print(B)
        # print('\n')
        # print(C)
        # print('\n')
        # print(np.dot(A, B))
        # print('\n')
        return C

    SubMatSize = matSize // blockSize
    for kk in range(0, SubMatSize * blockSize, blockSize):
        for jj in range(0, SubMatSize * blockSize, blockSize):
            for i in range(matSize):
                for j in range(jj, jj + blockSize):
                    for k in range(kk, kk + blockSize):
                        C[i, j] += A[i, k] * B[k, j]
    # print('Not 0 block')
    # print('\n')
    # print(A)
    # print('\n')
    # print(B)
    # print('\n')
    # print(C)
    # print('\n')
    # print(np.dot(A, B))
    # print('\n')
    return C


matSize = 1000
A = np.random.randint(0, 10, size=(matSize, matSize))
B = np.random.randint(0, 10, size=(matSize, matSize))
C = np.zeros((matSize, matSize))

print('with blocking')
tic = time.perf_counter()
BlockMatTest(matSize, 100, A, B, C)
toc = time.perf_counter()
print(toc - tic)

print('without blocking')
tic = time.perf_counter()
X = BlockMatTest(matSize, 0, A, B, C)
toc = time.perf_counter()
print(toc - tic)
# MatSizes = [2 ** n for n in range(3, 9)]
# series = dict()
# for matSize in MatSizes:
#     blockSizes = [min(2 ** n, matSize) for n in range(6)]
#     timeS = list()
#     blockSizeS = list()
#     for blockSize in blockSizes:
#         tic = time.perf_counter()
#         BlockMatTest(matSize, blockSize)
#         toc = time.perf_counter()
#         timeS.append(toc - tic)
#         blockSizeS.append(blockSize)
#     series[matSize] = (blockSizeS, timeS)
#
# for key in series:
#     X = series[key][0]
#     Y = series[key][1]
#     plt.plot(X, Y, label=key)
#
# plt.xlabel('Block Size')
# plt.ylabel('Time (s)')
# plt.title('Performance of Matrix Multiplication using blocking')
# plt.legend()
# plt.show()
