import numpy as np

from vennpy import set_logic


def generate_numeric_4():
    np.random.seed(42)
    pool = np.random.randint(0, 100, 250)
    i = 0
    A = set_logic.VSet("A", set(pool[i: + 70]))
    i += len(A)
    B = set_logic.VSet("B", set(pool[i:i + 100]))
    i += len(B)
    C = set_logic.VSet("C", set(pool[i:i + 30]))
    i += len(C)
    D = set_logic.VSet("D", set(pool[i:i + 50]))
    i += len(D)

    return A, B, C, D
