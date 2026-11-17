import numpy as np

from mlp import Camada, Rede, relu, sigmoid, softmax, tanh

inputsBatch = np.array([
    [0.5, 0.13,  0.9, 1],
    [2.1, 5.0, -1.0, 2],
    [3.1, 2.7,  3.3, 3]
])

rede = Rede([
    Camada([
        [ 0.6,  0.12, 0.6, 1.0],
        [-0.4,  0.18, 0.1, 3.1],
        [ 0.9, -0.12, 0.3, 2.1],
        [ 0.2,  0.5, -0.7, 0.4],
    ], [2, 3, -4, 1], relu),
    Camada([
        [ 0.3,  0.22, 0.4,  0.5],
        [-0.11, 0.78, 0.8, -0.2],
        [ 0.4, -0.62, 0.9,  0.1],
    ], [0, 1, -1], tanh),
    Camada([
        [ 0.2, -0.5,  0.3],
        [ 0.7,  0.4, -0.2],
        [-0.3,  0.8,  0.5],
    ], [0.1, -0.2, 0.3], sigmoid),
    Camada([
        [ 0.5, -0.3,  0.8],
        [-0.6,  0.9,  0.2],
    ], [0, 0.1], softmax),
])

print(rede)
saida = rede.forward(inputsBatch)
for i, camada in enumerate(rede.camadas, 1):
    print(f"Camada {i} ({camada.ativacao.__name__}):\n", camada.saida, camada.saida.shape)
print("Classe prevista por amostra:", rede.prever(inputsBatch))

np.random.seed(0)
redeAleatoria = Rede().adicionar(Camada.aleatoria(4, 5, relu)).adicionar(Camada.aleatoria(5, 2, softmax))
print("\n", redeAleatoria)
print(redeAleatoria.forward(inputsBatch))

assert saida.shape == (3, 2)
assert np.allclose(saida.sum(axis=1), 1)
assert np.allclose(saida, [[0.44060302, 0.55939698], [0.44045715, 0.55954285], [0.51720277, 0.48279723]])
try:
    Rede([Camada.aleatoria(4, 3), Camada.aleatoria(4, 2)])
    raise AssertionError("deveria recusar camadas que nao encaixam")
except ValueError:
    pass
