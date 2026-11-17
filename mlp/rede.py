import numpy as np

from mlp.ativacoes import linear


class Camada:
    def __init__(self, pesos, bias, ativacao=linear):
        self.pesos = np.array(pesos, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.ativacao = ativacao
        if self.pesos.shape[0] != self.bias.shape[0]:
            raise ValueError(f"pesos {self.pesos.shape} e bias {self.bias.shape} nao batem")

    @classmethod
    def aleatoria(cls, n_entradas, n_neuronios, ativacao=linear):
        return cls(0.1 * np.random.randn(n_neuronios, n_entradas), np.zeros(n_neuronios), ativacao)

    @property
    def n_entradas(self):
        return self.pesos.shape[1]

    @property
    def n_neuronios(self):
        return self.pesos.shape[0]

    def forward(self, entradas):
        self.entradas = np.array(entradas, dtype=float)
        self.z = np.dot(self.entradas, self.pesos.T) + self.bias
        self.saida = self.ativacao(self.z)
        return self.saida

    def __repr__(self):
        return f"Camada({self.n_entradas} -> {self.n_neuronios}, {self.ativacao.__name__})"


class Rede:
    def __init__(self, camadas=None):
        self.camadas = []
        for camada in camadas or []:
            self.adicionar(camada)

    def adicionar(self, camada):
        if self.camadas and self.camadas[-1].n_neuronios != camada.n_entradas:
            raise ValueError(f"{camada} nao encaixa depois de {self.camadas[-1]}")
        self.camadas.append(camada)
        return self

    def forward(self, entradas):
        saida = entradas
        for camada in self.camadas:
            saida = camada.forward(saida)
        return saida

    def prever(self, entradas):
        return np.argmax(self.forward(entradas), axis=1)

    def __repr__(self):
        return "Rede(\n  " + "\n  ".join(map(repr, self.camadas)) + "\n)"
