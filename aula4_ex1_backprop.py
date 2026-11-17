import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
np.random.seed(0)


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.pesos = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs, ativacao=None):
        self.inputs = inputs
        self.ativacao = ativacao
        self.z = np.dot(inputs, self.pesos) + self.bias
        if ativacao == "relu":
            self.output = self.relu(self.z)
        elif ativacao == "softmax":
            self.output = self.softmax(self.z)
        else:
            self.output = self.z

    def relu(self, inputs):
        return np.maximum(0, inputs)

    def softmax(self, inputs):
        saida = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return saida / np.sum(saida, axis=1, keepdims=True)

    def loss_categoricalCrossEntropy(self, chute, target):
        tentativas = len(chute)
        chute_ajustado = np.clip(chute, 1e-7, 1 - 1e-7)
        if len(target.shape) == 1:
            confianca = chute_ajustado[range(tentativas), target]
        elif len(target.shape) == 2:
            confianca = np.sum(chute_ajustado * target, axis=1)
        return -np.log(confianca)

    def loss(self, output, target):
        return np.mean(self.loss_categoricalCrossEntropy(output, target))

    def acuracia(self, targets):
        predicao = np.argmax(self.output, axis=1)
        acuracia = np.mean(predicao == targets)
        print(f"Acuracia = {acuracia}")

    def backward(self, dvalues=None, targets=None):
        if self.ativacao == "softmax":
            tentativas = len(self.output)
            if len(targets.shape) == 2:
                targets = np.argmax(targets, axis=1)
            dz = self.output.copy()
            dz[range(tentativas), targets] -= 1
            dz = dz / tentativas
        elif self.ativacao == "relu":
            dz = dvalues * (self.z > 0)
        else:
            dz = dvalues
        self.dpesos = np.dot(self.inputs.T, dz)
        self.dbias = np.sum(dz, axis=0, keepdims=True)
        self.dinputs = np.dot(dz, self.pesos.T)
        return self.dinputs


camada = Layer(len(digits.data[0]), 5)
camada2 = Layer(5, 15)
densa = Layer(15, 10)

inputs = digits.data[0:4]
targets = digits.target[0:4]


def forward_rede():
    camada.forward(inputs, "relu")
    camada2.forward(camada.output, "relu")
    densa.forward(camada2.output, "softmax")
    return densa.loss(densa.output, targets)


perda = forward_rede()
print("Perda media:", perda)
densa.acuracia(targets)

densa.backward(targets=targets)
camada2.backward(densa.dinputs)
camada.backward(camada2.dinputs)

for nome, c in [("camada", camada), ("camada2", camada2), ("densa", densa)]:
    print(f"\n{nome}: pesos {c.pesos.shape} -> dpesos {c.dpesos.shape} | bias {c.bias.shape} -> dbias {c.dbias.shape}")
    print("dbias:", np.round(c.dbias, 5))


def gradiente_numerico(parametro, eps=1e-5):
    grad = np.zeros_like(parametro)
    for i in np.ndindex(parametro.shape):
        original = parametro[i]
        parametro[i] = original + eps
        perda_mais = forward_rede()
        parametro[i] = original - eps
        perda_menos = forward_rede()
        parametro[i] = original
        grad[i] = (perda_mais - perda_menos) / (2 * eps)
    return grad


print("\nChecagem: backprop x gradiente numerico (diferenca relativa, quanto menor melhor)")
for nome, c in [("camada", camada), ("camada2", camada2), ("densa", densa)]:
    for param, grad in [("pesos", c.dpesos), ("bias", c.dbias)]:
        numerico = gradiente_numerico(getattr(c, param))
        erro = np.linalg.norm(grad - numerico) / (np.linalg.norm(grad) + np.linalg.norm(numerico) + 1e-12)
        print(f"{nome:8} {param:5}: {erro:.2e}")
        assert erro < 1e-6, f"{nome}.{param} com gradiente errado"
