import contextlib
import io

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import aula3mlp_na_mao as aula3

saida = aula3.densa.output
esperado = aula3.digits.target[0:len(saida)]
previsto = np.argmax(saida, axis=1)
confianca = np.max(saida, axis=1)

print(f"{'amostra':>7} | {'esperado':>8} | {'previsto':>8} | {'confianca':>9} | resultado")
for i, (e, p, c) in enumerate(zip(esperado, previsto, confianca)):
    print(f"{i:>7} | {e:>8} | {p:>8} | {c:>9.2%} | {'CORRETO' if e == p else 'ERRADO'}")

acertos = np.sum(previsto == esperado)
print(f"\nAcertos: {acertos}/{len(esperado)} ({acertos / len(esperado):.0%})")

print("\nProbabilidade de cada digito (linhas = amostras, colunas = digitos 0-9):")
print(np.round(saida, 4))

assert saida.shape == (4, 10)
assert np.allclose(saida.sum(axis=1), 1)
