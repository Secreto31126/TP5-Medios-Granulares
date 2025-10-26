from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np

FIT = Callable[[np.ndarray, float], np.ndarray]

def truncate_at_most_2(n: float) -> str:
    TRUNC = int(n * 100) / 100
    return str(TRUNC).rstrip('0').rstrip('.')

def sci_notation(val: float, _):
    if val == 0:
        return "0"

    EXP = int(np.floor(np.log10(abs(val))))
    COEFF = val / (10**EXP)

    return f"${truncate_at_most_2(COEFF)}\\times 10^{{{EXP}}}$"

def fitter(X: np.ndarray, Y: np.ndarray, F: FIT, LEFT: float, RIGHT: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    C = np.linspace(LEFT, RIGHT, 1000)
    E = np.array([np.sum((Y - F(X, c)) ** 2) for c in C])

    MIN = np.argmin(E)
    MIN_X = C[MIN]
    MIN_Y = E[MIN]

    return C, E, MIN_X, MIN_Y

def plot(C: np.ndarray, E: np.ndarray, MIN_X: float, MIN_Y: float, SCI: tuple[bool, bool] = (False, False)):
    plt.plot(C, E) # pyright: ignore[reportUnknownMemberType]

    label = rf'$a_{{\rm óptimo}} = ${sci_notation(MIN_X, 0)}'
    plt.axvline(x=MIN_X, color='r', linestyle='--', label=label) # pyright: ignore[reportUnknownMemberType]

    label = rf'$E(a_{{\rm óptimo}}) = ${sci_notation(MIN_Y, 0)}'
    plt.axhline(y=MIN_Y, color='g', linestyle='--', label=label) # pyright: ignore[reportUnknownMemberType]

    plt.xlabel('a', fontsize=24) # pyright: ignore[reportUnknownMemberType]
    plt.ylabel('E(a)', fontsize=24) # pyright: ignore[reportUnknownMemberType]


    if SCI[0]:
        plt.gca().xaxis.set_major_formatter(FuncFormatter(sci_notation)) # pyright: ignore[reportUnknownArgumentType]
        plt.xticks(fontsize=16) # pyright: ignore[reportUnknownMemberType]
    else:
        plt.xticks(fontsize=24) # pyright: ignore[reportUnknownMemberType]

    if SCI[1]:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(sci_notation)) # pyright: ignore[reportUnknownArgumentType]
        plt.yticks(fontsize=16) # pyright: ignore[reportUnknownMemberType]
    else:
        plt.yticks(fontsize=24) # pyright: ignore[reportUnknownMemberType]

    plt.legend(loc='upper center', bbox_to_anchor=(0.3, 1), fontsize=20) # pyright: ignore[reportUnknownMemberType]
    plt.show() # pyright: ignore[reportUnknownMemberType]

def origin(P1: tuple[float, float], P2: tuple[float, float]):
    M = (P2[1] - P1[1]) / (P2[0] - P1[0])
    return P1[1] - M * P1[0]

def LINEAR(B: float):
    def f(X: np.ndarray, c: float):
        return c * X + B
    return f
