import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

inputsBatch = np.array([
    [0.5, 0.13,  0.9, 1],
    [2.1, 5.0, -1.0, 2],
    [3.1, 2.7,  3.3, 3]
])

pesos1 = np.array([
    [ 0.6,  0.12, 0.6, 1.0],
    [-0.4,  0.18, 0.1, 3.1],
    [ 0.9, -0.12, 0.3, 2.1],
    [ 0.2,  0.5, -0.7, 0.4],
])
bias1 = np.array([2, 3, -4, 1])

pesos2 = np.array([
    [ 0.3,  0.22, 0.4,  0.5],
    [-0.11, 0.78, 0.8, -0.2],
    [ 0.4, -0.62, 0.9,  0.1],
])
bias2 = np.array([0, 1, -1])

pesos3 = np.array([
    [ 0.2, -0.5,  0.3],
    [ 0.7,  0.4, -0.2],
    [-0.3,  0.8,  0.5],
])
bias3 = np.array([0.1, -0.2, 0.3])

pesosSaida = np.array([
    [ 0.5, -0.3,  0.8],
    [-0.6,  0.9,  0.2],
])
biasSaida = np.array([0, 0.1])

z1 = np.dot(inputsBatch, pesos1.T) + bias1
a1 = relu(z1)
print("Camada 1 (ReLU):\n", a1, a1.shape)

z2 = np.dot(a1, pesos2.T) + bias2
a2 = tanh(z2)
print("Camada 2 (tanh):\n", a2, a2.shape)

z3 = np.dot(a2, pesos3.T) + bias3
a3 = sigmoid(z3)
print("Camada 3 - oculta adicional (sigmoid):\n", a3, a3.shape)

zSaida = np.dot(a3, pesosSaida.T) + biasSaida
saida = softmax(zSaida)
print("Saida (softmax) - probabilidade por classe:\n", saida, saida.shape)
print("Classe prevista por amostra:", np.argmax(saida, axis=1))

x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
print("\nx       :", x)
print("relu    :", relu(x))
print("sigmoid :", sigmoid(x).round(3))
print("tanh    :", tanh(x).round(3))
print("softmax :", softmax(x).round(3))

assert (relu(x) >= 0).all()
assert ((sigmoid(x) > 0) & (sigmoid(x) < 1)).all()
assert np.allclose(softmax(zSaida).sum(axis=1), 1)
assert saida.shape == (3, 2)
