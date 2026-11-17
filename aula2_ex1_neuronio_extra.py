import numpy as np

inputsBatch = np.array([
    [0.5, 0.13,  0.9, 1],
    [2.1, 5.0, -1.0, 2],
    [3.1, 2.7,  3.3, 3]
])  

pesos = np.array([
    [ 0.6,  0.12, 0.6, 1.0],
    [-0.4,  0.18, 0.1, 3.1],
    [ 0.9, -0.12, 0.3, 2.1],
    [ 0.2,  0.5, -0.7, 0.4],  
])  # 4x4
bias = np.array([2, 3, -4, 1])  

output1 = np.dot(inputsBatch, pesos.T) + bias
print("Camada oculta:\n", output1, output1.shape) 

input2 = output1
pesos2 = np.array([
    [ 0.3,  0.22, 0.4,  0.5],
    [-0.11, 0.78, 0.8, -0.2],
    [ 0.4, -0.62, 0.9,  0.1],
])  # 3x4 (era 3x3)
bias2 = np.array([0, 1, -1])

output2 = np.dot(input2, pesos2.T) + bias2
print("Camada 2:\n", output2, output2.shape) 

assert output1.shape == (3, 4)
assert output2.shape == (3, 3)
