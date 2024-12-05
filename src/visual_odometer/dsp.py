import numpy as np
from numpy import ndarray


# Frequency Windows:

def ideal_lowpass(I: ndarray, factor: float= .6) -> ndarray:
    m = factor * I.shape[0] / 2
    n = factor * I.shape[1] / 2
    N = np.min([m, n])
    I = I[int(I.shape[0] // 2 - N): int(I.shape[0] // 2 + N),
        int(I.shape[1] // 2 - N): int(I.shape[1] // 2 + N)]
    return I


# Spatial Windows:

def apply_raised_cosine_window(image: ndarray):
    rows, cols = image.shape
    i = np.arange(rows)
    j = np.arange(cols)
    window = 0.5 * (1 + np.cos(np.pi * (2 * i[:, None] - rows) / rows)) * \
             0.5 * (1 + np.cos(np.pi * (2 * j - cols) / cols))
    return image * window


def blackman_harris_window(size: int, a0: int, a1: int, a2: int, a3: int):
    # a0, a1, a2 e a3 são os coeficientes de janelamento
    # Criação do vetor de amostras
    n = np.arange(size)
    # Cálculo da janela de Blackman-Harris
    window = a0 - a1 * np.cos(2 * np.pi * n / (size - 1)) + a2 * np.cos(4 * np.pi * n / (size - 1)) - a3 * np.cos(
        6 * np.pi * n / (size - 1))
    return window


def apply_blackman_harris_window(image: ndarray,
                                 a0: int = 0.35875, a1: int = 0.48829, a2: int = 0.14128, a3: int = 0.01168):
    height, width = image.shape
    window_row = blackman_harris_window(width, a0, a1, a2, a3)
    window_col = blackman_harris_window(height, a0, a1, a2, a3)
    image_windowed = np.outer(window_col, window_row) * image
    return image_windowed
